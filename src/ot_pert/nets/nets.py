from collections.abc import Sequence
from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from ott.neural.networks.layers import time_encoder
from ott.neural.networks.velocity_field import VelocityField


def get_masks(dataset: List[jnp.ndarray]):
    attention_mask = []
    for data in dataset:
        if data.ndim < 2:
            data = data[None, :]
        if data.ndim < 3:
            data = data[None, :]
        mask = jnp.all(data == 0.0, axis=-1)
        mask = 1 - mask
        mask = jnp.outer(mask, mask)
        attention_mask.append(mask)
    return jnp.expand_dims(jnp.equal(jnp.array(attention_mask), 1.0), 1)


class CondVelocityField(nn.Module):
  r"""Neural vector field.

  This class learns a map :math:`v: \mathbb{R}\times \mathbb{R}^d
  \rightarrow \mathbb{R}^d` solving the ODE :math:`\frac{dx}{dt} = v(t, x)`.
  Given a source distribution at time :math:`t_0`, the velocity field can be
  used to transport the source distribution given at :math:`t_0` to
  a target distribution given at :math:`t_1` by integrating :math:`v(t, x)`
  from :math:`t=t_0` to :math:`t=t_1`.

  Args:
    hidden_dims: Dimensionality of the embedding of the data.
    output_dims: Dimensionality of the embedding of the output.
    condition_dims: Dimensionality of the embedding of the condition.
      If :obj:`None`, the velocity field has no conditions.
    time_dims: Dimensionality of the time embedding.
      If :obj:`None`, ``hidden_dims`` is used.
    time_encoder: Time encoder for the velocity field.
    act_fn: Activation function.
  """
  hidden_dims: Sequence[int]
  output_dims: Sequence[int]
  condition_dims: Optional[Sequence[int]] = None
  time_dims: Optional[Sequence[int]] = None
  time_encoder: Callable[[jnp.ndarray],
                         jnp.ndarray] = time_encoder.cyclical_time_encoder
  act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(
      self,
      t: jnp.ndarray,
      x: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      train: bool = True,
      return_embedding: bool = False,
  ) -> jnp.ndarray:
    """Forward pass through the neural vector field.

    Args:
      t: Time of shape ``[batch, 1]``.
      x: Data of shape ``[batch, ...]``.
      condition: Conditioning vector of shape ``[batch, ...]``.
      train: If `True`, enables dropout for training.

    Returns:
      Output of the neural vector field of shape ``[batch, output_dim]``.
    """
    time_dims = self.hidden_dims if self.time_dims is None else self.time_dims

    t = self.time_encoder(t)
    for time_dim in time_dims:
      t = self.act_fn(nn.Dense(time_dim)(t))
      t = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(t)

    for hidden_dim in self.hidden_dims:
      x = self.act_fn(nn.Dense(hidden_dim)(x))
      x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

    if self.condition_dims is not None:
      assert condition is not None, "No condition was passed."
      for cond_dim in self.condition_dims:
        condition = self.act_fn(nn.Dense(cond_dim)(condition))
        condition = nn.Dropout(
            rate=self.dropout_rate, deterministic=not train
        )(
            condition
        )
      if return_embedding:
        return condition
      feats = jnp.concatenate([t, x, condition], axis=-1)
    else:
      feats = jnp.concatenate([t, x], axis=-1)

    for output_dim in self.output_dims[:-1]:
      feats = self.act_fn(nn.Dense(output_dim)(feats))
      feats = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(feats)

    # No activation function for the final layer
    return nn.Dense(self.output_dims[-1])(feats)

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
      cond = jnp.ones((1, condition_dim))

    params = self.init(rng, t, x, cond, train=False)["params"]
    return train_state.TrainState.create(
        apply_fn=self.apply, params=params, tx=optimizer)

  def get_embedding(self, vf_state: train_state.TrainState, condition: jnp.ndarray):
    """Get the embedding of `condition`"""
    return self.apply({"params": vf_state.params}, jnp.zeros((len(condition), 1)), jnp.zeros((len(condition), self.output_dims[-1])), condition, train=False, return_embedding=True)

class VelocityFieldWithAttention(nn.Module):
    num_heads: int
    qkv_feature_dim: int
    max_seq_length: int
    hidden_dims: Sequence[int]
    output_dims: Sequence[int]
    condition_dims: Optional[Sequence[int]] = None
    time_dims: Optional[Sequence[int]] = None
    time_encoder: Callable[[jnp.ndarray], jnp.ndarray] = time_encoder.cyclical_time_encoder
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    dropout_rate: float = 0.0
   
    def __post_init__(self):
        self.get_masks = jax.jit(get_masks)
        super().__post_init__()

    @nn.compact
    def __call__(
        self,
        t: jnp.ndarray,
        x: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        train: bool = True,
        return_embedding: bool = False,
    ) -> jnp.ndarray:
        """Forward pass through the neural vector field.

        Args:
          t: Time of shape ``[batch, 1]``.
          x: Data of shape ``[batch, ...]``.
          condition: Conditioning vector of shape ``[batch, ...]``.

        Returns
        -------
          Output of the neural vector field of shape ``[batch, output_dim]``.
        """
        squeeze_output = False
        if x.ndim < 2:
            x = x[None, :]
            t = jnp.full(shape=(1, 1), fill_value=t)
            condition = condition[None, :]
            squeeze_output = True

        time_dims = self.hidden_dims if self.time_dims is None else self.time_dims
        t = self.time_encoder(t)
        for time_dim in time_dims:
            t = self.act_fn(nn.Dense(time_dim)(t))
            t = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(t)

        for hidden_dim in self.hidden_dims:
            x = self.act_fn(nn.Dense(hidden_dim)(x))
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        assert condition is not None, "No condition sequence was passed."

        token_shape = (len(condition), 1) if condition.ndim > 2 else (1,)
        class_token = nn.Embed(num_embeddings=1, features=condition.shape[-1])(jnp.int32(jnp.zeros(token_shape)))

        condition = jnp.concatenate((class_token, condition), axis=-2)
        mask = self.get_masks(condition)

        attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.qkv_feature_dim, dropout_rate=self.dropout_rate, deterministic=not train)
        emb = attention(condition, mask=mask)
        condition = emb[:, 0, :]  # only continue with token 0

        for cond_dim in self.condition_dims:
            condition = self.act_fn(nn.Dense(cond_dim)(condition))
            condition = nn.Dropout(
            rate=self.dropout_rate, deterministic=not train
        )(
            condition
        )
        if return_embedding:
           return condition

        feats = jnp.concatenate([t, x, condition], axis=1)

        for output_dim in self.output_dims[:-1]:
            feats = self.act_fn(nn.Dense(output_dim)(feats))
            feats = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(feats)

        # no activation function for the final layer
        out = nn.Dense(self.output_dims[-1])(feats)
        return jnp.squeeze(out) if squeeze_output else out
    
    def get_embedding(self,
        vf_state: train_state.TrainState,
        condition: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get the embedding of `condition`.

        Args:
            condition: Condition vector

        Returns:
            embedding of `condition`. 
        """
        
        return self.apply({"params": vf_state.params}, jnp.zeros((len(condition), 1)), jnp.zeros((len(condition), self.output_dims[-1])), condition, train=False, return_embedding=True)


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

        Returns
        -------
          The training state.
        """
        t, x = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        if self.condition_dims is None:
            cond = None
        else:
            assert condition_dim > 0, "Condition dimension must be positive."
            cond = jnp.ones((1, 1, condition_dim))

        params = self.init(rng, t, x, cond, train=False)["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class GENOTVelocityFieldWithAttention(nn.Module):
    split_dim: int
    num_heads: int
    qkv_feature_dim: int
    max_seq_length: int
    hidden_dims: Sequence[int]
    output_dims: Sequence[int]
    condition_dims: Sequence[int]
    condition_dims_forward: Sequence[int]
    condition_dims_post_attention: Sequence[int]
    time_dims: Optional[Sequence[int]] = None
    time_encoder: Callable[[jnp.ndarray], jnp.ndarray] = time_encoder.cyclical_time_encoder
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    dropout_rate: float = 0.0
  
    def __post_init__(self):
        self.get_masks = jax.jit(get_masks)
        super().__post_init__()

    @nn.compact
    def __call__(
        self,
        t: jnp.ndarray,
        x: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        train: bool = True,
        return_embedding: bool = False,
    ) -> jnp.ndarray:
        """Forward pass through the neural vector field.

        Args:
          t: Time of shape ``[batch, 1]``.
          x: Data of shape ``[batch, ...]``.
          condition: Conditioning vector of shape ``[batch, ...]``.

        Returns
        -------
          Output of the neural vector field of shape ``[batch, output_dim]``.
        """
        squeeze_output = False
        if x.ndim < 2:
            x = x[None, :]
            t = jnp.full(shape=(1, 1), fill_value=t)
            condition = condition[None, :]
            squeeze_output = True

        time_dims = self.hidden_dims if self.time_dims is None else self.time_dims
        t = self.time_encoder(t)
        for time_dim in time_dims:
            t = self.act_fn(nn.Dense(time_dim)(t))
            t = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(t)

        for hidden_dim in self.hidden_dims:
            x = self.act_fn(nn.Dense(hidden_dim)(x))
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        assert condition is not None, "No condition sequence was passed."
        condition_forward = condition[:, 0, : self.split_dim]  # the first split_dim elements are the source data points
        condition_attention = condition[..., self.split_dim :]  # the remaining elements are conditions

        for cond_dim in self.condition_dims_forward:
            condition_forward = self.act_fn(nn.Dense(cond_dim)(condition_forward))
            condition_forward = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(condition_forward)

        token_shape = (len(condition_attention), 1) if condition_attention.ndim > 2 else (1,)
        class_token = nn.Embed(num_embeddings=1, features=condition_attention.shape[-1])(
            jnp.int32(jnp.zeros(token_shape))
        )

        condition_attention = jnp.concatenate((class_token, condition_attention), axis=-2)
        mask = self.get_masks(condition_attention)

        attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.qkv_feature_dim, dropout_rate=self.dropout_rate, deterministic=not train)
        emb = attention(condition_attention, mask=mask)
        condition = emb[:, 0, :]  # only continue with token 0
        for cond_dim in self.condition_dims_post_attention:
            condition = self.act_fn(nn.Dense(cond_dim)(condition))
            condition = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(condition)
            
        if return_embedding:
           return condition

        cond_all = jnp.concatenate((condition_forward, condition), axis=1)
        for cond_dim in self.condition_dims:
            condition = self.act_fn(nn.Dense(cond_dim)(cond_all))
            condition = nn.Dropout(
            rate=self.dropout_rate, deterministic=not train
            )(
                condition
            )
        if return_embedding:
          return condition

        feats = jnp.concatenate([t, x, condition], axis=1)

        for output_dim in self.output_dims[:-1]:
            feats = self.act_fn(nn.Dense(output_dim)(feats))
            feats = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(feats)

        # no activation function for the final layer
        out = nn.Dense(self.output_dims[-1])(feats)
        return jnp.squeeze(out) if squeeze_output else out
    
    def get_embedding(self,
        vf_state: train_state.TrainState,
        condition: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Get embedding of first part of `condition`.
        
        Note that `condition` consists of [`cond`, `source_data_point`]
        concatenated along axis 1. In the function only `cond` will be 
        used, but the input is expected to consist of the concatenated
        vector for the sake of uniformness. 

        Args:
            condition: Condition vector.

        Returns:
            The embedding of `cond`.
        """
        return self.apply({"params": vf_state.params}, jnp.zeros((len(condition), 1)), jnp.zeros((len(condition), self.output_dims[-1])), condition, train=False, return_embedding=True)


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

        Returns
        -------
          The training state.
        """
        t, x = jnp.ones((1, 1)), jnp.ones((1, input_dim))
        if self.condition_dims is None:
            cond = None
        else:
            assert condition_dim > 0, "Condition dimension must be positive."
            cond = jnp.ones((1, 1, condition_dim))

        params = self.init(rng, t, x, cond, train=False)["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
