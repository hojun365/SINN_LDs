import jax
import math
import jax.numpy as jnp
from flax import linen as nn
pytorch_init = nn.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform")

# ══════════════════════════════════════════════════════════════════════════════
# Fully connected neural network (FCN)
# ══════════════════════════════════════════════════════════════════════════════
class Linear(nn.Module):
    n_hid: int
    @nn.compact
    def __call__(self, X):
        X = jnp.tanh(nn.Dense(self.n_hid, use_bias=False, kernel_init=pytorch_init, name="lin1")(X))
        X = jnp.tanh(nn.Dense(self.n_hid, use_bias=False, kernel_init=pytorch_init, name="lin2")(X))
        return X

class FCN(nn.Module):
    N: int
    n_hid: int
    n_block: int

    @nn.compact
    def __call__(self, x, y, bias):
        X = jnp.concatenate([x, y, bias], axis=-1)
        X = jnp.tanh(nn.Dense(self.n_hid, use_bias=False, kernel_init=pytorch_init, name="encoder")(X))
        for n in range(self.n_block): X = X + Linear(self.n_hid, name=f"body_{n}")(X)
        X = nn.Dense(2 * self.N, use_bias=False, kernel_init=pytorch_init, name="decoder")(X)
        return (X[..., :self.N], X[..., self.N:])

# ══════════════════════════════════════════════════════════════════════════════
# Deepsets
# ══════════════════════════════════════════════════════════════════════════════
class Deepset(nn.Module):
    n_hid: int
    use_lambda: bool = False

    @nn.compact
    def __call__(self, X, lam=None):
        lin1_out = nn.Dense(self.n_hid, use_bias=False, kernel_init=pytorch_init, name="lin1")(X)
        lin2_out = nn.Dense(self.n_hid, use_bias=False, kernel_init=pytorch_init, name="lin2")(jnp.mean(X, axis=-2, keepdims=True))
        out = lin1_out - lin2_out
        
        if self.use_lambda:
            W_lambda = self.param("W_lambda", nn.initializers.normal(stddev=self.n_hid ** -0.5), (self.n_hid,))
            if lam is not None: out = out + lam * W_lambda
        return out

class Ritz(nn.Module):
    n_hid: int
    @nn.compact
    def __call__(self, X):
        X = jnp.tanh(Deepset(self.n_hid, name="lin1")(X))
        X = jnp.tanh(Deepset(self.n_hid, name="lin2")(X))
        return X

class EQI(nn.Module):
    N: int
    n_hid: int
    n_block: int

    @nn.compact
    def __call__(self, x, y, lam):
        X = jnp.concatenate([x[..., None], y[..., None]], axis=-1)
        X = jnp.tanh(Deepset(self.n_hid, use_lambda=True, name="encoder")(X, lam[..., None]))
        for n in range(self.n_block): X = X + Ritz(self.n_hid, name=f"body_{n}")(X)
        X = Deepset(2, name="decoder")(X)
        return (X[..., 0], X[..., 1])