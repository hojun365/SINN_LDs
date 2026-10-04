# ═════════════════════════════════════════════════════════════════════════
# Library
# ═════════════════════════════════════════════════════════════════════════
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import math
import jax
jax.config.update("jax_enable_x64", True)
import optax
import pickle
import argparse
import numpy as np
import Model as MM
import jax.numpy as jnp
from tqdm import trange
from functools import partial
DT = jnp.float32

# ═════════════════════════════════════════════════════════════════════════
# Parser
# ═════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--N",     type=int,   default=40)
parser.add_argument("--hid",   type=int,   default=80)
parser.add_argument("--block", type=int,   default=4)
parser.add_argument("--lmbda", type=float, default=0.0)
parser.add_argument("--lr",    type=float, default=1e-2)
parser.add_argument("--epoch", type=int,   default=30000)
args = parser.parse_args()

N        = args.N
lr       = args.lr
n_hid    = args.hid
lmbda    = args.lmbda
epochs   = args.epoch
n_blocks = args.block

os.makedirs("result/", exist_ok=True)
suffix = f"N={N}_H={n_hid}_B={n_blocks}_λ={lmbda}"

# ═════════════════════════════════════════════════════════════════════════
# Physical & numerical parameters
# ═════════════════════════════════════════════════════════════════════════
n_models   = 100
mid_epochs = epochs // 20
n_shift    = N - 1
epsilon    = 1.0
Dt         = 1.0
Dr         = 3 * Dt
v          = 10.0
π          = math.pi
ρ          = 0.1
L          = math.sqrt(N / ρ) + 1e-5
ensemble   = 75
B          = 2 * ensemble
dt         = 0.0001
tstep      = 1000

inv_entr = v / Dt
inv_girs = 1.0 / (4 * Dt)
inv_om   = 1.0 / (4 * Dt * dt)

lmbda0 = lmbda
lmbda1 = -0.9

Vr = math.sqrt(2 * dt * Dt)
Vθ = math.sqrt(2 * dt * Dr)

# ═════════════════════════════════════════════════════════════════════════
# Initial state & Global Tensors
# ═════════════════════════════════════════════════════════════════════════
key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), "big"))
key, kθ, kx, ky, kinit = jax.random.split(key, 5)

θ = 2 * π * jax.random.uniform(kθ, (B, N), dtype=DT)
x = (L * jax.random.uniform(kx, (B, N), dtype=DT)) % L
y = (L * jax.random.uniform(ky, (B, N), dtype=DT)) % L

bias_train = jnp.concatenate([lmbda0 * jnp.ones((ensemble, 1), dtype=DT),
                              lmbda1 * jnp.ones((ensemble, 1), dtype=DT)], axis=0)
bias_eval  = lmbda0 * jnp.ones((B, 1), dtype=DT)
corrector  = jnp.asarray(2**(1/3), DT) * jnp.eye(N, dtype=DT)

# ═════════════════════════════════════════════════════════════════════════
# Network & Optimizer
# ═════════════════════════════════════════════════════════════════════════
NN = MM.FCN(N, n_hid, n_blocks)
params = NN.init(kinit, x, y, bias_train)
params = jax.tree_util.tree_map(lambda p: p.astype(DT), params)

Optimizer = optax.chain(optax.add_decayed_weights(1e-3), optax.scale_by_adam(), optax.scale(-lr))
opt_state = Optimizer.init(params)

# ═════════════════════════════════════════════════════════════════════════
# Functions
# ═════════════════════════════════════════════════════════════════════════
def PBC(X):
    return X % L

def MIC(d):
    return (d + L / 2.0) % L - L / 2.0

def Shift(x, y):
    return (MIC(x - x.mean(axis=-1, keepdims=True)), MIC(y - y.mean(axis=-1, keepdims=True)))

def Force(x, y, θ):
    dx = MIC(x[..., :, None] - x[..., None, :])
    dy = MIC(y[..., :, None] - y[..., None, :])
    l_ij = dx**2 + dy**2 + corrector
    ff = jnp.maximum(l_ij**(-7.0) - 0.5 * l_ij**(-4.0), 0.0)
    Fx = 48 * epsilon * (ff * dx).sum(-1) + v * jnp.cos(θ)
    Fy = 48 * epsilon * (ff * dy).sum(-1) + v * jnp.sin(θ)
    return Fx, Fy

def Roll_idx(n):
    ar = jnp.arange(N)
    return (ar - n) % N, (ar + n) % N

def Roll_n_Unroll(params, x, y, bias, f_idx, i_idx):
    Ox, Oy = NN.apply(params, x[..., f_idx], y[..., f_idx], bias)
    return Ox[..., i_idx], Oy[..., i_idx]

def Measure(params, sx, sy, FNx, FNy, bias):
    def scan_fn(err, s):
        f_idx, i_idx = Roll_idx(s)
        rx, ry = Roll_n_Unroll(params, sx, sy, bias, f_idx, i_idx)
        return err + jnp.mean((FNx - rx)**2) + jnp.mean((FNy - ry)**2), None
    total, _ = jax.lax.scan(scan_fn, jnp.zeros((), DT), jnp.arange(1, n_shift + 1))
    return total / n_shift

# ═════════════════════════════════════════════════════════════════════════
# Simulate
# ═════════════════════════════════════════════════════════════════════════
def simulate_fn(params, x, y, θ, bias, key, steps, measure=False):
    norm = dt * steps

    def step_fn(carry, _):
        x, y, θ, entr, girs, mt, err, k = carry
        k, k1, k2, k3 = jax.random.split(k, 4)
        dWθ = jax.random.normal(k1, (B, N), dtype=DT) * Vθ
        dWx = jax.random.normal(k2, (B, N), dtype=DT) * Vr
        dWy = jax.random.normal(k3, (B, N), dtype=DT) * Vr

        sx, sy   = Shift(x, y)
        F0x, F0y = Force(x, y, θ)
        FNx, FNy = NN.apply(params, sx, sy, bias)

        dx = dt * (F0x + FNx) + dWx
        dy = dt * (F0y + FNy) + dWy

        girs_next = girs + dt * inv_girs * (FNx**2 + FNy**2).sum(-1)
        entr_next = entr + inv_entr * (jnp.cos(θ) * dx + jnp.sin(θ) * dy).sum(-1)
        mt_next   = mt + inv_om * (dWx**2 + dWy**2).sum(-1)

        sg = jax.lax.stop_gradient
        err_next = err + dt * Measure(params, sx, sy, sg(FNx), sg(FNy), bias) if measure else err

        θ_next = (θ + dWθ) % (2 * jnp.pi)
        x_next = PBC(x + sg(dx))
        y_next = PBC(y + sg(dy))
        return (x_next, y_next, θ_next, entr_next, girs_next, mt_next, err_next, k), None

    step_ckpt = jax.checkpoint(step_fn, prevent_cse=False)

    z = jnp.zeros(B, dtype=DT)
    init = (x, y, θ, z, z, z, jnp.zeros((), DT), key)
    (xf, yf, θf, entr_f, girs_f, mt_f, err_f, _), _ = jax.lax.scan(step_ckpt, init, None, length=steps)

    Error = err_f / norm
    loss  = (girs_f - bias[:, 0] * entr_f).mean() / norm
    return xf, yf, θf, loss, entr_f, mt_f, Error

# ═════════════════════════════════════════════════════════════════════════
# JIT steps
# ═════════════════════════════════════════════════════════════════════════
@jax.jit
def train_step(params, opt_state, x, y, θ, key):
    key, ks = jax.random.split(key)

    def loss_fn(p):
        xf, yf, θf, loss, _, mt, _ = simulate_fn(p, x, y, θ, bias_train, ks, tstep, measure=False)
        return loss, (xf, yf, θf, mt)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    xf, yf, θf, mt = aux
    updates, opt_state = Optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, xf, yf, θf, loss, mt, key

@partial(jax.jit, static_argnames=('steps', 'measure'))
def eval_step(params, x, y, θ, key, steps, measure=False):
    key, ks = jax.random.split(key)
    xf, yf, θf, loss, entr, _, err = simulate_fn(params, x, y, θ, bias_eval, ks, steps, measure=measure)
    return xf, yf, θf, loss, entr, err, key

# ═════════════════════════════════════════════════════════════════════════
# Replica exchange
# ═════════════════════════════════════════════════════════════════════════
@jax.jit
def Replica(x, y, θ, M_T, key):
    key, k0, k1, ku = jax.random.split(key, 4)
    g0 = jax.random.permutation(k0, ensemble)
    g1 = ensemble + jax.random.permutation(k1, ensemble)

    M0 = M_T[g0].astype(jnp.float64)
    M1 = M_T[g1].astype(jnp.float64)
    log_alpha = -(M0 - M1)

    U = jax.random.uniform(ku, (ensemble,), dtype=jnp.float64)
    swaps = U < jnp.exp(log_alpha)
    sw = swaps[:, None]

    def swap(A):
        a0, a1 = A[g0], A[g1]
        return A.at[g0].set(jnp.where(sw, a1, a0)).at[g1].set(jnp.where(sw, a0, a1))

    return swap(x), swap(y), swap(θ), swaps.mean(), key

# ═════════════════════════════════════════════════════════════════════════
# Results
# ═════════════════════════════════════════════════════════════════════════
SCGF, ENTR, ERROR = [], [], []

# ═════════════════════════════════════════════════════════════════════════
# Warm-up & 0 epoch
# ═════════════════════════════════════════════════════════════════════════
x, y, θ, _, _, err, key = eval_step(params, x, y, θ, key, 2 * tstep, measure=True)
ERROR.append(float(err))

# ═════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════
for epoch in trange(epochs):
    params, opt_state, x, y, θ, loss, M_T, key = train_step(params, opt_state, x, y, θ, key)
    x, y, θ, acc, key = Replica(x, y, θ, M_T, key)

    loss_val = float(loss)
    if not math.isfinite(loss_val): raise RuntimeError("Diverge")

    if (epoch + 1) % mid_epochs == 0:
        *_, err, key = eval_step(params, x, y, θ, key, 2 * tstep, measure=True)
        ERROR.append(float(err))

    if epoch >= (epochs - n_models):
        _, _, _, loss_e, entr_e, _, key = eval_step(params, x, y, θ, key, 50 * tstep, measure=False)
        SCGF.append(-float(loss_e) / N)
        ENTR.append(float(entr_e.mean()) / (N * dt * 50 * tstep))

# ═════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════
for name, obj in [("ERROR", ERROR), ("SCGF", SCGF), ("ENTR", ENTR)]:
    with open(f"result/FCN_{name}_{suffix}.pkl", "wb") as f: pickle.dump(obj, f)

xf, yf, θf, *_ = eval_step(params, x, y, θ, key, 50 * tstep, measure=False)
np.savez(f"result/FCN_STATE_{suffix}.npz", x=np.asarray(xf), y=np.asarray(yf), θ=np.asarray(θf))