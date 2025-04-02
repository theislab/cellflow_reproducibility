from typing import Callable, Optional, Sequence, Tuple, Union

import optax
from flax import linen as nn
from flax.training import train_state
from sklearn.metrics import r2_score
from cfp.data._dataloader import PredictionSampler, TrainSampler, ValidationSampler
import itertools


import jax.numpy as jnp
from ott.neural.networks.layers.posdef import PositiveDense
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple, Iterable, Type, Mapping

from collections import defaultdict, abc
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, Type
import pandas as pd
import anndata as ad
import cfp.preprocessing as cfpp

import optax
from flax.core import freeze
from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs
from ott.geometry.pointcloud import PointCloud

import cfp
import scanpy as sc
import numpy as np
import functools
from ott.solvers import utils as solver_utils
import optax
from omegaconf import OmegaConf
from typing import NamedTuple, Any

from cfp.data._data import ConditionData, ValidationData
from cfp.data._dataloader import PredictionSampler, TrainSampler, ValidationSampler




class ICNN(nn.Module):
    """Input convex neural network (ICNN) architecture."""

    dim_hidden: Sequence[int]
    input_dim: int
    cond_dim: int
    init_std: float = 0.1
    init_fn: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] = nn.initializers.normal  # type: ignore[name-defined]  # noqa: E501
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu  # type: ignore[name-defined]
    pos_weights: bool = False

    def setup(self):
        """Initialize ICNN architecture."""
        num_hidden = len(self.dim_hidden)

        Dense = PositiveDense if self.pos_weights else nn.Dense
        kernel_inits_wz = [self.init_fn(self.init_std) for _ in range(num_hidden + 1)]

        w_xs = []
        w_zs = []
        for i in range(0, num_hidden):
            w_xs.append(
                nn.Dense(
                    self.dim_hidden[i],
                    kernel_init=self.init_fn(self.init_std),
                    bias_init=self.init_fn(self.init_std),
                    use_bias=True,
                )
            )
            if i != 0:
                w_zs.append(
                    Dense(
                        self.dim_hidden[i],
                        kernel_init=kernel_inits_wz[i],
                        use_bias=False,
                    )
                )
        w_xs.append(
            nn.Dense(
                1,
                kernel_init=self.init_fn(self.init_std),
                bias_init=self.init_fn(self.init_std),
                use_bias=True,
            )
        )
        w_zs.append(Dense(1, kernel_init=kernel_inits_wz[-1], use_bias=False))
        self.w_xs = w_xs
        self.w_zs = w_zs

        if self.cond_dim:
            w_zu = []
            w_xu = []
            w_u = []
            v = []

            for i in range(0, num_hidden):
                if i != 0:
                    w_zu.append(
                        nn.Dense(
                            self.dim_hidden[i],
                            kernel_init=self.init_fn(self.init_std),
                            use_bias=True,
                            bias_init=self.init_fn(self.init_std),
                        )
                    )
                w_xu.append(  # this the matrix that multiply with x
                    nn.Dense(
                        self.input_dim,  # self.dim_hidden[i],
                        kernel_init=self.init_fn(self.init_std),
                        use_bias=True,
                        bias_init=self.init_fn(self.init_std),
                    )
                )
                w_u.append(
                    nn.Dense(
                        self.dim_hidden[i],
                        kernel_init=self.init_fn(self.init_std),
                        use_bias=True,
                        bias_init=self.init_fn(self.init_std),
                    )
                )
                v.append(
                    nn.Dense(
                        2,
                        kernel_init=self.init_fn(self.init_std),
                        use_bias=True,
                        bias_init=self.init_fn(self.init_std),
                    )
                )
            w_zu.append(
                nn.Dense(
                    self.dim_hidden[-1],
                    kernel_init=self.init_fn(self.init_std),
                    use_bias=True,
                    bias_init=self.init_fn(self.init_std),
                )
            )
            w_xu.append(  # this the matrix that multiply with x
                nn.Dense(
                    self.input_dim,
                    kernel_init=self.init_fn(self.init_std),
                    use_bias=True,
                    bias_init=self.init_fn(self.init_std),
                )
            )
            w_u.append(
                nn.Dense(
                    1,
                    kernel_init=self.init_fn(self.init_std),
                    bias_init=self.init_fn(self.init_std),
                    use_bias=True,
                )
            )
            v.append(
                nn.Dense(
                    1,
                    kernel_init=self.init_fn(self.init_std),
                    bias_init=self.init_fn(self.init_std),
                    use_bias=True,
                )
            )

            self.w_zu = w_zu
            self.w_xu = w_xu
            self.w_u = w_u
            self.v = v

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: Optional[jnp.ndarray] = None) -> jnp.ndarray:  # type: ignore[name-defined]
        """Apply ICNN module."""
        assert (c is not None) == (self.cond_dim > 0), "`conditional` flag and whether `c` is provided must match."

        if not self.cond_dim:
            z = self.w_xs[0](x)
            z = jnp.multiply(z, z)
            for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
                z = self.act_fn(jnp.add(Wz(z), Wx(x)))
            y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x))
        else:
            # Initialize
            mlp_condition_embedding = self.w_xu[0](c)
            x_hadamard_1 = jnp.multiply(x, mlp_condition_embedding)
            mlp_condition = self.w_u[0](c)
            z = jnp.add(mlp_condition, self.w_xs[0](x_hadamard_1))
            z = jnp.multiply(z, z)
            u = self.act_fn(self.v[0](c))

            for Wz, Wx, Wzu, Wxu, Wu, V in zip(
                self.w_zs[:-1], self.w_xs[:-1], self.w_zu[:-1], self.w_xu[1:-1], self.w_u[1:-1], self.v[1:-1]
            ):
                mlp_convex = jnp.clip(Wzu(u), a_min=0)
                z_hadamard_1 = jnp.multiply(z, mlp_convex)
                mlp_condition_embedding = Wxu(u)
                x_hadamard_1 = jnp.multiply(x, mlp_condition_embedding)
                mlp_condition = Wu(u)
                z = self.act_fn(jnp.add(jnp.add(Wz(z_hadamard_1), Wx(x_hadamard_1)), mlp_condition))
                u = self.act_fn(V(u))

            mlp_convex = jnp.clip(self.w_zu[-1](u), a_min=0)  # bs x d
            z_hadamard_1 = jnp.multiply(z, mlp_convex)  # bs x d

            mlp_condition_embedding = self.w_xu[-1](u)  # bs x d
            x_hadamard_1 = jnp.multiply(x, mlp_condition_embedding)  # bs x d

            mlp_condition = self.w_u[-1](u)
            y = jnp.add(jnp.add(self.w_zs[-1](z_hadamard_1), self.w_xs[-1](x_hadamard_1)), mlp_condition)

        return jnp.squeeze(y, axis=-1)

    def create_train_state(
        self,
        rng: jnp.ndarray,  # type: ignore[name-defined]
        optimizer: optax.OptState,
        input_shape: Union[int, Tuple[int, ...]],
    ) -> train_state.TrainState:
        """Create initial `TrainState`."""
        condition = (
            jnp.ones(
                shape=[
                    self.cond_dim,
                ]
            )
            if self.cond_dim
            else None
        )
        params = self.init(rng, x=jnp.ones(input_shape), c=condition)["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
    





Train_t = Dict[str, Dict[str, Union[float, List[float]]]]


def _get_icnn(
    input_dim: int,
    cond_dim: int,
    pos_weights: bool = False,
    dim_hidden: Iterable[int] = (64, 64, 64, 64),
    **kwargs: Any,
) -> ICNN:
    return ICNN(input_dim=input_dim, cond_dim=cond_dim, pos_weights=pos_weights, dim_hidden=dim_hidden, **kwargs)


def _get_optimizer(
    learning_rate: float = 1e-4, b1: float = 0.5, b2: float = 0.9, weight_decay: float = 0.0, **kwargs: Any
) -> Type[optax.GradientTransformation]:
    return optax.adamw(learning_rate=learning_rate, b1=b1, b2=b2, weight_decay=weight_decay, **kwargs)







class OTTNeuralDualSolver:
    """Solver of the ICNN-based Kantorovich dual.

    Optimal transport mapping via input convex neural networks,
    Makkuva-Taghvaei-Lee-Oh, ICML'20.
    http://proceedings.mlr.press/v119/makkuva20a/makkuva20a.pdf

    Parameters
    ----------
    input_dim
        Input dimension of data (without condition)
    conditional
        Whether to use partial input convex neural networks (:cite:`bunne2022supervised`).
    batch_size
        Batch size.
    tau_a
        Unbalancedness parameter in the source distribution in the inner sampling loop.
    tau_b
        Unbalancedness parameter in the target distribution in the inner sampling loop.
    epsilon
        Entropic regularisation parameter in the inner sampling loop.
    seed
        Seed for splitting the data.
    pos_weights
        If `True` enforces non-negativity of corresponding weights of ICNNs, else only penalizes negativity.
    dim_hidden
        The length of `dim_hidden` determines the depth of the ICNNs, while the entries of the list determine
        the layer widhts.
    beta
        If `pos_weights` is not `None`, this determines the multiplicative constant of L2-penalization of
        negative weights in ICNNs.
    best_model_metric
        Which metric to use to assess model training. The specified metric needs to be computed in the passed
        `callback_func`. By default `sinkhorn_loss_forward` only takes into account the error in the forward map,
        while `sinkhorn` computes the mean error between the forward and the inverse map.
    iterations
        Number of (outer) training steps (batches) of the training process.
    inner_iters
        Number of inner iterations for updating the convex conjugate.
    valid_freq
        Frequency at which the model is evaluated.
    log_freq
        Frequency at which training is logged.
    patience
        Number of iterations of no performance increase after which to apply early stopping.
    optimizer_f_kwargs
        Keyword arguments for the optimizer :class:`optax.adamw` for f.
    optimizer_g_kwargs
        Keyword arguments for the optimizer :class:`optax.adamw` for g.
    pretrain_iters
        Number of iterations (batches) for pretraining with the identity map.
    pretrain_scale
        Variance of Gaussian distribution used for pretraining.
    sinkhorn_kwargs
        Keyword arguments for computing the discrete sinkhorn divergence for assessing model training.
        By default, the same `tau_a`, `tau_b` and `epsilon` are taken as for the inner sampling loop.
    compute_wasserstein_baseline
        Whether to compute the Sinkhorn divergence between the source and the target distribution as
        a baseline for the Wasserstein-2 distance computed with the neural solver.
    callback_func
        Callback function to compute metrics during training. The function takes as input the
        target and source batch and the predicted target and source batch and returns a dictionary of
        metrics.

    Warning
    -------
    If `compute_wasserstein_distance` is `True`, a discrete OT problem has to be solved on the validation
    dataset which scales linearly in the validation set size. If `train_size=1.0` the validation dataset size
    is the full dataset size, hence this is a source of prolonged run time or Out of Memory Error.
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int = 0,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
        seed: int = 0,
        pos_weights: bool = False,
        f: Union[Dict[str, Any], ICNN] = MappingProxyType({}),
        g: Union[Dict[str, Any], ICNN] = MappingProxyType({}),
        beta: float = 1.0,
        iterations: int = 25000,  # TODO(@MUCDK): rename to max_iterations
        inner_iters: int = 10,
        valid_freq: int = 250,
        log_freq: int = 10,
        patience: int = 100,
        optimizer_f: Union[Dict[str, Any], Type[optax.GradientTransformation]] = MappingProxyType({}),
        optimizer_g: Union[Dict[str, Any], Type[optax.GradientTransformation]] = MappingProxyType({}),
        pretrain_iters: int = 15001,
        pretrain_scale: float = 3.0,
        valid_sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        compute_wasserstein_baseline: bool = True,
        callback_func: Optional[
            Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], Dict[str, float]]
        ] = None,
    ):
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.batch_size = batch_size
        self.tau_a = 1.0 if tau_a is None else tau_a
        self.tau_b = 1.0 if tau_b is None else tau_b
        self.epsilon = epsilon if self.tau_a != 1.0 or self.tau_b != 1.0 else None
        self.pos_weights = pos_weights
        self.beta = beta
        self.iterations = iterations
        self.inner_iters = inner_iters
        self.valid_freq = valid_freq
        self.log_freq = log_freq
        self.patience = patience
        self.pretrain_iters = pretrain_iters
        self.pretrain_scale = pretrain_scale
        self.key: jax.random.PRNGKeyArray = jax.random.PRNGKey(seed)

        self.optimizer_f = _get_optimizer(**optimizer_f) if isinstance(optimizer_f, abc.Mapping) else optimizer_f
        self.optimizer_g = _get_optimizer(**optimizer_g) if isinstance(optimizer_g, abc.Mapping) else optimizer_g
        self.neural_f = _get_icnn(input_dim=input_dim, cond_dim=cond_dim, **f) if isinstance(f, abc.Mapping) else f
        self.neural_g = _get_icnn(input_dim=input_dim, cond_dim=cond_dim, **g) if isinstance(g, abc.Mapping) else g
        self.callback_func = callback_func
        
        # set optimizer and networks
        self.setup(self.neural_f, self.neural_g, self.optimizer_f, self.optimizer_g)

    def setup(self, neural_f: ICNN, neural_g: ICNN, optimizer_f: optax.OptState, optimizer_g: optax.OptState):
        """Initialize all components required to train the :class:`moscot.backends.ott.NeuralDual`.

        Parameters
        ----------
        neural_f
            Network to parameterize the forward transport map.
        neural_g
            Network to parameterize the reverse transport map.
        optimizer_f
            Optimizer for `neural_f`.
        optimizer_g
            Optimizer for `neural_g`.
        """
        key_f, key_g, self.key = jax.random.split(self.key, 3)  # type:ignore[arg-type]

        # check setting of network architectures
        if neural_g.pos_weights != self.pos_weights or neural_f.pos_weights != self.pos_weights:
            logger.warning(
                f"Setting of ICNN and the positive weights setting of the \
                      `NeuralDualSolver` are not consistent. Proceeding with \
                      the `NeuralDualSolver` setting, with positive weigths \
                      being {self.pos_weights}."
            )
            neural_g.pos_weights = self.pos_weights
            neural_f.pos_weights = self.pos_weights

        self.state_f = neural_f.create_train_state(key_f, optimizer_f, self.input_dim)
        self.state_g = neural_g.create_train_state(key_g, optimizer_g, self.input_dim)

        self.train_step_f = self.get_train_step(to_optimize="f")
        self.train_step_g = self.get_train_step(to_optimize="g")


    def __call__(
        self,
        trainloader,
        N_PCs: int,
    ) -> Any:
        """Start the training pipeline of the :class:`moscot.backends.ott.NeuralDual`.

        Parameters
        ----------
        trainloader
            Data loader for the training data.

        Returns
        -------
        The trained model and training statistics.
        """
        pretrain_logs = {}
        if self.pretrain_iters > 0:
            condition_arr = jnp.squeeze(jnp.concatenate((train_dataloader._data.condition_data["drugs"], train_dataloader._data.condition_data["dose"], train_dataloader._data.condition_data["cell_line"]), axis=-1))
            pretrain_logs = self.pretrain_identity(condition_arr)

        self.train_neuraldual(trainloader, N_PCs)
    
    
        return None

    def pretrain_identity(
        self, conditions: Optional[jnp.ndarray]  # type:ignore[name-defined]
    ) -> Train_t:  # TODO(@lucaeyr) conditions can be `None` right?
        """Pretrain the neural networks to parameterize the identity map.

        Parameters
        ----------
        conditions
            Conditions in the case of a conditional Neural OT model, otherwise `None`.

        Returns
        -------
        Pre-training statistics.
        """

        def pretrain_loss_fn(
            params: jnp.ndarray,  # type: ignore[name-defined]
            data: jnp.ndarray,  # type: ignore[name-defined]
            condition: jnp.ndarray,  # type: ignore[name-defined]
            state: TrainState,
        ) -> float:
            """Loss function for the pretraining on identity."""
            grad_g_data = jax.vmap(jax.grad(lambda x: state.apply_fn({"params": params}, x, condition), argnums=0))(
                data
            )
            # loss is L2 reconstruction of the input
            return ((grad_g_data - data) ** 2).sum(axis=1).mean()  # TODO make nicer

        @jax.jit
        def pretrain_update(
            state: TrainState, key: jax.Array
        ) -> Tuple[jnp.ndarray, TrainState]:  # type:ignore[name-defined]
            """Update function for the pretraining on identity."""
            # sample gaussian data with given scale
            x = self.pretrain_scale * jax.random.normal(key, [self.batch_size, self.input_dim])
            condition = jax.random.choice(key, conditions) if self.cond_dim else None  # type:ignore[arg-type]
            grad_fn = jax.value_and_grad(pretrain_loss_fn, argnums=0)
            loss, grads = grad_fn(state.params, x, condition, state)
            return loss, state.apply_gradients(grads=grads)

        pretrain_logs: Dict[str, List[float]] = {"loss": []}
        for iteration in tqdm(range(self.pretrain_iters)):
            key_pre, self.key = jax.random.split(self.key, 2)  # type:ignore[arg-type]
            # train step for potential g directly updating the train state
            loss, self.state_g = pretrain_update(self.state_g, key_pre)
            # clip weights of g
            if not self.pos_weights:
                self.state_g = self.state_g.replace(params=self.clip_weights_icnn(self.state_g.params))
            if iteration % self.log_freq == 0:
                pretrain_logs["loss"].append(loss)
        # load params of g into state_f
        # this only works when f & g have the same architecture
        self.state_f = self.state_f.replace(params=self.state_g.params)
        return {"pretrain_logs": pretrain_logs}  # type:ignore[dict-item]

    def train_neuraldual(
        self,
        dataloader,
        N_PCs,
    ) -> Train_t:
        """Train the model.

        Parameters
        ----------
        trainloader
            Data loader for the training data.

        Returns
        -------
        Training statistics.
        """
        # set logging dictionaries
        train_logs: Dict[str, List[float]] = defaultdict(list)
        valid_logs: Dict[str, Union[List[float], float]] = defaultdict(list)
        sink_dist: List[float] = []
        curr_patience: int = 0
        best_loss: float = jnp.inf
        best_iter_distance: float = jnp.inf
        best_params_f: jnp.ndarray = self.state_f.params
        best_params_g: jnp.ndarray = self.state_g.params  # type:ignore[name-defined]

        # define dict to contain source and target batch
        batch: Dict[str, jnp.ndarray] = {}  # type:ignore[name-defined]
        
        for iteration in tqdm(range(self.iterations)):
            # sample policy and condition if given in trainloader
            key, self.key = jax.random.split(self.key, 2)  # type:ignore[arg-type]
            batch = dataloader.sample(key)
            
            batch["source"]=batch["src_cell_data"][:,:N_PCs]
            batch["target"]= batch["tgt_cell_data"][:,:N_PCs]
            condition_arr = jnp.concatenate((train_dataloader._data.condition_data["drugs"], train_dataloader._data.condition_data["dose"], train_dataloader._data.condition_data["cell_line"]), axis=-1)
            batch["condition"] = jnp.squeeze(condition_arr[0,...])

           
            self.state_f, train_f_metrics = self.train_step_f(self.state_f, self.state_g, batch)
            self.state_g, train_g_metrics = self.train_step_g(self.state_f, self.state_g, batch)
            if not self.pos_weights:
                self.state_g = self.state_g.replace(params=self.clip_weights_icnn(self.state_g.params))
            
    def get_train_step(
        self,
        to_optimize: Literal["f", "g"],
    ) -> Callable[  # type:ignore[name-defined]
        [TrainState, TrainState, Dict[str, jnp.ndarray]], Tuple[TrainState, Dict[str, float]]
    ]:
        """Get one training step."""

        def loss_f_fn(
            params_f: jnp.ndarray,  # type:ignore[name-defined]
            params_g: jnp.ndarray,  # type:ignore[name-defined]
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],  # type:ignore[name-defined]
        ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:  # type:ignore[name-defined]
            """Loss function for f."""
            # get loss terms of kantorovich dual
            grad_f_src = jax.vmap(
                jax.grad(lambda x: state_f.apply_fn({"params": params_f}, x, batch["condition"]), argnums=0)
            )(batch["source"])
            g_grad_f_src = jax.vmap(lambda x: state_g.apply_fn({"params": params_g}, x, batch["condition"]))(grad_f_src)
            src_dot_grad_f_src = jnp.sum(batch["source"] * grad_f_src, axis=1)
            # compute loss
            loss = jnp.mean(g_grad_f_src - src_dot_grad_f_src)
            if not self.pos_weights:
                penalty = self.beta * self.penalize_weights_icnn(params_f)
                loss += penalty
            else:
                penalty = 0
            return loss, [penalty]

        def loss_g_fn(
            params_f: jnp.ndarray,  # type:ignore[name-defined]
            params_g: jnp.ndarray,  # type:ignore[name-defined]
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],  # type:ignore[name-defined]
        ) -> Tuple[jnp.ndarray, List[float]]:  # type: ignore[name-defined]
            """Loss function for g."""
            # get loss terms of kantorovich dual
            grad_f_src = jax.vmap(
                jax.grad(lambda x: state_f.apply_fn({"params": params_f}, x, batch["condition"]), argnums=0)
            )(batch["source"])
            g_grad_f_src = jax.vmap(lambda x: state_g.apply_fn({"params": params_g}, x, batch["condition"]))(grad_f_src)
            src_dot_grad_f_src = jnp.sum(batch["source"] * grad_f_src, axis=1)
            # compute loss
            g_tgt = jax.vmap(lambda x: state_g.apply_fn({"params": params_g}, x, batch["condition"]))(batch["target"])
            loss = jnp.mean(g_tgt - g_grad_f_src)
            total_loss = jnp.mean(g_grad_f_src - g_tgt - src_dot_grad_f_src)
            # compute wasserstein distance
            dist = 2 * total_loss + jnp.mean(
                jnp.sum(batch["target"] * batch["target"], axis=1)
                + 0.5 * jnp.sum(batch["source"] * batch["source"], axis=1)
            )
            return loss, [total_loss, dist]

        @jax.jit
        def step_fn(
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],  # type: ignore[name-defined]
        ) -> Tuple[TrainState, Dict[str, float]]:
            """Step function for training."""
            # get loss function for f or g
            if to_optimize == "f":
                grad_fn = jax.value_and_grad(loss_f_fn, argnums=0, has_aux=True)
                # compute loss, gradients and metrics
                (loss, raw_metrics), grads = grad_fn(state_f.params, state_g.params, state_f, state_g, batch)
                # return updated state and metrics dict
                metrics = {"loss_f": loss, "penalty": raw_metrics[0]}
                return state_f.apply_gradients(grads=grads), metrics
            if to_optimize == "g":
                grad_fn = jax.value_and_grad(loss_g_fn, argnums=1, has_aux=True)
                # compute loss, gradients and metrics
                (loss, raw_metrics), grads = grad_fn(state_f.params, state_g.params, state_f, state_g, batch)
                # return updated state and metrics dict
                metrics = {"loss_g": loss, "loss": raw_metrics[0], "w_dist": raw_metrics[1]}
                return state_g.apply_gradients(grads=grads), metrics
            raise NotImplementedError()

        return step_fn


    def clip_weights_icnn(self, params: FrozenVariableDict) -> FrozenVariableDict:
        """Clip weights of ICNN."""
        for key in params:
            if key.startswith("w_zs"):
                params[key]["kernel"] = jnp.clip(params[key]["kernel"], a_min=0)

        return params

    def penalize_weights_icnn(self, params: FrozenVariableDict) -> float:
        """Penalize weights of ICNN."""
        penalty = 0
        for key in params:
            if key.startswith("w_z"):
                penalty += jnp.linalg.norm(jax.nn.relu(-params[key]["kernel"]))
        return penalty


    @property
    def is_balanced(self) -> bool:
        """Return whether the problem is balanced."""
        return self.tau_a == self.tau_b == 1.0

def split_by_first_last_underscore(s):
    last_underscore = s.rfind('_')
    second_last_underscore = s[:last_underscore].rfind('_')
    
    if last_underscore == -1 or second_last_underscore == -1:
        return s, None, None  # If there are less than two underscores
    
    first_part = s[:second_last_underscore]
    middle_part = s[second_last_underscore + 1:last_underscore]
    last_part = s[last_underscore + 1:]
    
    return first_part, middle_part, last_part
    
def get_condition(x):
    drug, dose, cell_line = split_by_first_last_underscore(x["condition"])
    return cell_line+"_"+drug+"_"+str(float(10**int(float(dose))))
    
    

split=5
adata_train_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_train_{split}.h5ad"
adata_test_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_test_{split}.h5ad"
adata_ood_path = f"/lustre/groups/ml01/workspace/ot_perturbation/data/sciplex/adata_ood_{split}.h5ad"
adata_train = sc.read_h5ad(adata_train_path)
adata_test = sc.read_h5ad(adata_test_path)
adata_ood = sc.read_h5ad(adata_ood_path)

adata_train.uns["sample_covariates_one_hot"] = {"A549": 0, "K562": 1, "MCF7": 2}
adata_test.uns["sample_covariates_one_hot"] = {"A549": 0, "K562": 1, "MCF7": 2}
adata_ood.uns["sample_covariates_one_hot"] = {"A549": 0, "K562": 1, "MCF7": 2}
cf = cfp.model.CellFlow(adata_train, solver="otfm")


cf.prepare_data(
    sample_rep="X_pca",
    control_key="control",
    perturbation_covariates={"drugs": ["drug"], "dose": ["logdose"]},
    perturbation_covariate_reps={"drugs": "ecfp_dict"},
    sample_covariates = ["cell_line"],
    sample_covariate_reps = {"cell_line": "sample_covariates_one_hot"},
    split_covariates=["cell_line"],
)


cf.prepare_validation_data(
    adata_test,
    name="test",
    n_conditions_on_log_iteration=None,
    n_conditions_on_train_end=None,
)



cf.prepare_validation_data(
    adata_ood,
    name="ood",
    n_conditions_on_log_iteration=None,
    n_conditions_on_train_end=None,
)



validation_loaders = {
    k: ValidationSampler(v) for k, v in cf.validation_data.items()
}


n_pcs = [50, 100, 300]
pretrain_iters = [0, 1000, 10_000]
iterations = [50_000, 200_000]
batch_sizes = [256, 1024]


combinations_dicts = [
    {
        'N_PCs': N_PCs,
        'pretrain_iters': pre_iters,
        'iterations': iters,
        'batch_size': bs,
        
    }
    for N_PCs, pre_iters, iters, bs in itertools.product(
        n_pcs,
        pretrain_iters,
        iterations,
        batch_sizes,
    )
]
df_res = pd.DataFrame(columns=['N_PCs', 'pretrain_iters', 'iterations', 'batch_size', 'r2_ood'])


for i,cd in enumerate(combinations_dicts):
    N_PCs = cd["N_PCs"]
    pretrain_iters = cd["pretrain_iters"]
    iterations = cd["iterations"]
    batch_size = cd["batch_size"]

    train_dataloader = TrainSampler(data=cf.train_data, batch_size=batch_size)
    
    solver = OTTNeuralDualSolver(input_dim=N_PCs, cond_dim=1026, pretrain_iters=pretrain_iters, iterations=iterations, batch_size=batch_size)
    solver(train_dataloader, N_PCs=N_PCs)
    out = validation_loaders["ood"].sample(mode="on_train_end")
    batch = {}
    preds_ood = {}
    source_dict = out["source"]
    condition_dict = out["condition"]
    target_dict = out["target"]
    for cond in source_dict.keys():
        batch["source"]=source_dict[cond][:,:N_PCs]
        batch["target"]= target_dict[cond][:,:N_PCs]
        condition_arr = jnp.concatenate((condition_dict[cond]["drugs"], condition_dict[cond]["dose"], condition_dict[cond]["cell_line"]), axis=-1)
        batch["condition"] = jnp.squeeze(condition_arr[0,...])

        preds_ood[cond] = jax.vmap(
                    jax.grad(lambda x: solver.state_f.apply_fn({"params": solver.state_f.params}, x, batch["condition"]), argnums=0)
                )(batch["source"])


    adapted_preds_ood = {k[0]+"_"+str(k[1])+"_"+k[2]: v for k,v in preds_ood.items()}

    all_data = []
    conditions = []

    for condition, array in adapted_preds_ood.items():
        all_data.append(array)
        conditions.extend([condition] * array.shape[0])
        
    # Stack all data vertically to create a single array
    all_data_array = np.vstack(all_data)

    # Create a DataFrame for the .obs attribute
    obs_data = pd.DataFrame({
        'condition': conditions
    })

    # Create the Anndata object
    adata_ood_result = ad.AnnData(X=np.empty((len(all_data_array),adata_train.n_vars)), obs=obs_data)
    adata_ood_result.obsm["X_pca_pred"] = np.concatenate((all_data_array, np.zeros((len(all_data_array), 300-N_PCs))), axis=1)
    cfpp.reconstruct_pca(query_adata=adata_ood_result, use_rep="X_pca_pred", ref_adata=adata_train, layers_key_added="X_recon_pred")
    adata_ood_result.obs["condition_adapted"] = adata_ood_result.obs.apply(get_condition, axis=1)



    r2_scores = {}
    for cond in adata_ood.obs["condition"].cat.categories:
        if "Vehicle" in cond:
            continue
        true = adata_ood[adata_ood.obs["condition"]==cond].X.toarray()
        pred = adata_ood_result[adata_ood_result.obs["condition_adapted"]==cond].layers["X_recon_pred"]
        r2_scores[cond] = r2_score(true.mean(axis=0), pred.mean(axis=0))
    r2 = np.mean(list(r2_scores.values()))

    df_res.loc[i] = list(cd.values()) + [r2]
    df_res.to_csv("results_hsearch.csv")