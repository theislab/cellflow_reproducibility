from typing import Callable, Optional, Sequence, List

import jax
import jax.numpy as jnp

import optax
from flax import linen as nn
from flax.training import train_state

from ott.neural.networks.layers import time_encoder
import functools

def get_masks(dataset: List[jnp.ndarray], max_seq_length: int, pad_max_dim: Optional[int] = None, pad_token=0):
    # dataset should be of size [batch_size, max_seq_length, dim_concatenated_conditions]
    # and the first `pad_max_dim` dimensions of `dim_concatenated_conditions` should contain 0.0
    # if an element of the sequence is considered to be `None`
    attention_mask = []
    for data in dataset:
        if data.ndim<2:
            data = data[None, :]
        if data.ndim<3:
            data = data[None, :]
        mask = jnp.all(data[:, :pad_max_dim, 0] == 0.0, axis=1)
        mask = 1-mask
        mask = jnp.outer(mask, mask)
        attention_mask.append(mask)
    return jnp.expand_dims(jnp.array(attention_mask), 1)



class VelocityFieldWithAttention(nn.Module):
    num_heads: int
    qkv_feature_dim: int
    max_seq_length: int
    hidden_dims: Sequence[int]
    output_dims: Sequence[int]
    condition_dims: Optional[Sequence[int]] = None
    time_dims: Optional[Sequence[int]] = None
    time_encoder: Callable[[jnp.ndarray],
                         jnp.ndarray] = time_encoder.cyclical_time_encoder
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    pad_max_dim: int = -1

    def __post_init__(self):
        self.get_masks = jax.jit(functools.partial(get_masks, max_seq_length=self.max_seq_length+1, pad_max_dim=self.pad_max_dim))
        super().__post_init__()


    @nn.compact
    def __call__(
      self,
      t: jnp.ndarray,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Forward pass through the neural vector field.
        
        Args:
          t: Time of shape ``[batch, 1]``.
          x: Data of shape ``[batch, ...]``.
          condition: Conditioning vector of shape ``[batch, ...]``.
        
        Returns:
          Output of the neural vector field of shape ``[batch, output_dim]``.
        """
        squeeze_output = False
        if x.ndim < 2:
            x = x[None,:]
            t = jnp.full(shape=(1, 1), fill_value=t)
            condition = condition[None,:]
            squeeze_output = True
            
        time_dims = self.hidden_dims if self.time_dims is None else self.time_dims
        t = self.time_encoder(t)
        for time_dim in time_dims:
          t = self.act_fn(nn.Dense(time_dim)(t))
        
        for hidden_dim in self.hidden_dims:
          x = self.act_fn(nn.Dense(hidden_dim)(x))
    
        assert condition is not None, "No condition sequence was passed."

        token_shape = (len(condition),1) if condition.ndim > 2 else (1,)
        class_token = nn.Embed(num_embeddings=1, features=condition.shape[-1])(jnp.int32(jnp.zeros(token_shape)))
        
        condition = jnp.concatenate((class_token, condition), axis=-2)
        mask = self.get_masks(condition) 
            
        attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.qkv_feature_dim)
        emb = attention(condition, mask=mask)
        emb = emb[:,0,:] # only continue with token 0
        
        for cond_dim in self.condition_dims:
            condition = self.act_fn(nn.Dense(cond_dim)(emb))
    
        
        feats = jnp.concatenate([t, x, condition], axis=1)
        
        for output_dim in self.output_dims[:-1]:
          feats = self.act_fn(nn.Dense(output_dim)(feats))
        
        # no activation function for the final layer
        out =  nn.Dense(self.output_dims[-1])(feats)
        return jnp.squeeze(out) if squeeze_output else out

    def create_train_state(
          self,
          rng: jax.Array,
          optimizer: optax.OptState,
          input_dim: int,
          condition_dim: Optional[int] = None,
      ) -> train_state.TrainState:
        """Create the training state.
    
        Args:
          rng: Random number generator.
          optimizer: Optimizer.
          input_dim: Dimensionality of the velocity field.
          condition_dim: Dimensionality of the condition of the velocity field.
    
        Returns:
          The training state.
        """
        t, x = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        if self.condition_dims is None:
          cond = None
        else:
          assert condition_dim > 0, "Condition dimension must be positive."
          cond = jnp.ones((1, 1, condition_dim))
    
        params = self.init(rng, t, x, cond)["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply, params=params, tx=optimizer
        )