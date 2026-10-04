import jax
import math
import jax.numpy as jnp
from flax import linen as nn

# ═════════════════════════════════════════════════════════════════════════
# Initializer
# ═════════════════════════════════════════════════════════════════════════
def pytorch_init(key, shape, dtype=jnp.float32):
    bound = shape[-2] ** -0.5
    return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=+bound)

def stacked_pytorch_init(key, shape, dtype=jnp.float32):
    # fan-in is N * dim_in : the einsum contracts both j and c
    keys  = jax.random.split(key, shape[0])
    bound = (shape[0] * shape[1]) ** -0.5
    return jax.vmap(lambda k: jax.random.uniform(k, shape[1:], dtype=dtype,
                                                 minval=-bound, maxval=+bound))(keys)

def scaled_stacked_init(input_scale):
    def init(key, shape, dtype=jnp.float32):
        W = stacked_pytorch_init(key, shape, dtype)
        s = jnp.asarray(input_scale, dtype)[None, :, None]      # (1, dim_in, 1)
        return W / s
    return init

def scaled_dense_init(input_scale):
    def init(key, shape, dtype=jnp.float32):
        W = pytorch_init(key, shape, dtype)
        s = jnp.asarray(input_scale, dtype)[:, None]            # (fan_in, 1)
        return W / s
    return init

# ═════════════════════════════════════════════════════════════════════════
# Fully connected network
# ═════════════════════════════════════════════════════════════════════════
class Ritz(nn.Module):
    n_hid: int

    @nn.compact
    def __call__(self, X):
        X = jnp.tanh(nn.Dense(self.n_hid, use_bias=False, kernel_init=pytorch_init)(X))
        X = jnp.tanh(nn.Dense(self.n_hid, use_bias=False, kernel_init=pytorch_init)(X))
        return X

class FCN(nn.Module):
    N: int
    n_hid: int
    n_block: int

    @nn.compact
    def __call__(self, g, a, λ):
        X = jnp.concatenate((g, a, jnp.full((g.shape[0], 1), jnp.asarray(λ, dtype=g.dtype), dtype=g.dtype)), axis=-1)
        scale = jnp.concatenate([jnp.full((self.N,), float(self.N)), jnp.ones((self.N + 1,))])
        X = nn.Dense(self.n_hid, use_bias=False, kernel_init=scaled_dense_init(scale))(X)
        for n in range(self.n_block): X = X + Ritz(self.n_hid)(X)
        out = nn.Dense(2 * self.N, use_bias=False, kernel_init=pytorch_init)(X)
        return out[..., :self.N], out[..., self.N:]

# ═════════════════════════════════════════════════════════════════════════
# Equivariant mixer
# ═════════════════════════════════════════════════════════════════════════
class Mixer(nn.Module):
    N: int
    dim_in: int
    dim_out: int
    use_λ: bool = False
    input_scale: tuple = ()

    def setup(self):
        init = (scaled_stacked_init(self.input_scale) if self.input_scale else stacked_pytorch_init)
        self.M = self.param("M", init, (self.N, self.dim_in, self.dim_out))
        if self.use_λ: self.W_λ = self.param("W_λ", nn.initializers.normal(stddev=self.dim_out ** -0.5), (self.dim_out,))

    def build_W(self):
        idx = (jnp.arange(self.N)[None, :] - jnp.arange(self.N)[:, None]) % self.N
        return self.M[idx]

    def __call__(self, X, W=None, λ=None):
        W = self.build_W() if W is None else W
        out = jnp.einsum("ijco,bjc->bio", W, X)
        if self.use_λ: out = out + λ * self.W_λ
        return out

# ═════════════════════════════════════════════════════════════════════════
# Equivariant residual block
# ═════════════════════════════════════════════════════════════════════════
class RitzMixer(nn.Module):
    N: int
    dim: int

    def setup(self):
        self.mix1 = Mixer(N=self.N, dim_in=self.dim, dim_out=self.dim, name="mix1")
        self.mix2 = Mixer(N=self.N, dim_in=self.dim, dim_out=self.dim, name="mix2")

    def build_Ws(self):
        return (self.mix1.build_W(), self.mix2.build_W())

    def __call__(self, X, Ws=None):
        W1, W2 = self.build_Ws() if Ws is None else Ws
        X = jnp.tanh(self.mix1(X, W1))
        X = jnp.tanh(self.mix2(X, W2))
        return X

# ═════════════════════════════════════════════════════════════════════════
# Equivariant network
# ═════════════════════════════════════════════════════════════════════════
class EQI(nn.Module):
    N: int
    n_hid: int
    n_block: int

    def setup(self):
        self.encoder = Mixer(N=self.N, dim_in=2, dim_out=self.n_hid, use_λ=True, input_scale=(float(self.N), 1.0))
        self.body = [RitzMixer(N=self.N, dim=self.n_hid) for n in range(self.n_block)]
        self.decoder = Mixer(N=self.N, dim_in=self.n_hid, dim_out=2)

    def build_Ws(self):
        return (self.encoder.build_W(), tuple(block.build_Ws() for block in self.body), self.decoder.build_W())

    def __call__(self, g, a, λ, Ws=None):
        W_encoder, W_body, W_decoder = self.build_Ws() if Ws is None else Ws
        X = jnp.stack((g, a), axis=-1)
        X = self.encoder(X, W_encoder, λ)
        for n, block in enumerate(self.body): X = X + block(X, W_body[n])
        out = self.decoder(X, W_decoder)
        return out[..., 0], out[..., 1]