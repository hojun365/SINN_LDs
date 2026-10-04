# ═════════════════════════════════════════════════════════════════════════
# Library
# ═════════════════════════════════════════════════════════════════════════
import os
import jax
import math
import optax
import pickle
import argparse
import numpy as np
import Model as MM
import jax.numpy as jnp
from tqdm import trange
from functools import partial

# ═════════════════════════════════════════════════════════════════════════
# Parser arguments
# ═════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--N",     type=int,   required=True)
parser.add_argument("--hid",   type=int,   required=True)
parser.add_argument("--block", type=int,   required=True)
parser.add_argument("--lr",    type=float, default =1e-2)
parser.add_argument("--lmbda", type=float, required=True)
parser.add_argument("--epoch", type=int,   default =20000)
parser.add_argument("--dt",    type=float, default =0.001)
parser.add_argument("--tstep", type=int,   default =5000)
args = parser.parse_args()

N        = args.N
lr       = args.lr
n_hid    = args.hid
lmbda    = args.lmbda
epochs   = args.epoch
n_blocks = args.block

RESULT_DIR  = "./result"
DIVERGE_DIR = "./Diverge"
os.makedirs(RESULT_DIR,  exist_ok=True)
os.makedirs(DIVERGE_DIR, exist_ok=True)
suffix = f"N={N}_H={n_hid}_B={n_blocks}_λ={lmbda}"

# ═════════════════════════════════════════════════════════════════════════
# Physical & numerical parameters
# ═════════════════════════════════════════════════════════════════════════
B        = 200
n_models = 100
tstep    = args.tstep
dt       = args.dt
mid_epochs = epochs // 50
L = 1.0; μ = 1.0; τ = 1.0; K = 10.0; Da = 2.0; kbT = 0.5

inv_work = 1 / μ
inv_gpos = 1 / (4 * μ * kbT)
inv_gvel = 1 / (4 * Da / (τ ** 2))

# ═════════════════════════════════════════════════════════════════════════
# Periodic boundary
# ═════════════════════════════════════════════════════════════════════════
GAP = jnp.zeros((1, N)).at[0, -1].set(+L)
PBC = jnp.zeros((1, N)).at[0, +0].set(-L).at[0, -1].set(+L)

# ═════════════════════════════════════════════════════════════════════════
# Diffusion coefficient
# ═════════════════════════════════════════════════════════════════════════
Vr = math.sqrt(2 * dt * μ * kbT)
Va = math.sqrt(2 * dt * Da / (τ ** 2))

# ═════════════════════════════════════════════════════════════════════════
# Initial state
# ═════════════════════════════════════════════════════════════════════════
key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), "big"))
key, key_model, key_r, key_a = jax.random.split(key, 4)

a = math.sqrt(Da / τ) * jax.random.normal(key_a, (B, N))
r = L * jnp.sort(jax.random.uniform(key_r, (B, N)), axis=-1)

# ═════════════════════════════════════════════════════════════════════════
# Network & Optimizer
# ═════════════════════════════════════════════════════════════════════════
NN = MM.FCN(N, n_hid, n_blocks)
params = NN.init(key_model, r, a, lmbda)

Optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.scale(-lr))
opt_state = Optimizer.init(params)

# ═════════════════════════════════════════════════════════════════════════
# ERROR Measure Function
# ═════════════════════════════════════════════════════════════════════════
def Roll_idx(n):
    ar = jnp.arange(N)
    return (ar - n) % N, (ar + n) % N

def Roll_n_Unroll(params, r, a, f_idx, i_idx):
    Or, Oa = NN.apply(params, r[..., f_idx], a[..., f_idx], lmbda)
    return Or[..., i_idx], Oa[..., i_idx]

def Measure(params, r, a):
    FNr, FNa = NN.apply(params, r, a, lmbda)
    den = jnp.array([jnp.mean(FNr ** 2), jnp.mean(FNa ** 2)])

    def scan_fn(num, s):
        f_idx, i_idx = Roll_idx(s)
        FNr_s, FNa_s = Roll_n_Unroll(params, r, a, f_idx, i_idx)
        e_r = jnp.mean((FNr - FNr_s) ** 2)
        e_a = jnp.mean((FNa - FNa_s) ** 2)
        return num + jnp.array([e_r, e_a]), None

    num, _ = jax.lax.scan(scan_fn, jnp.zeros(2), jnp.arange(1, N))
    return jnp.concatenate([num / (N - 1), den])

# ═════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════
def loss_fn(params, r, a, key, steps, measure=False):
    norm = dt * steps

    @partial(jax.checkpoint, prevent_cse=False, static_argnums=())
    def step_fn(carry, _):
        r, a, acc, k = carry
        k, kr, ka = jax.random.split(k, 3)
        dWr = jax.random.normal(kr, (B, N)) * Vr
        dWa = jax.random.normal(ka, (B, N)) * Va

        g = (jnp.roll(r, -1, axis=-1) - r + GAP) * N
        FNr, FNa = NN.apply(params, g, a, lmbda)

        Fa = -a / τ + FNa
        Fr = μ * K * (jnp.roll(r, -1, axis=-1) - 2*r + jnp.roll(r, +1, axis=-1) + PBC) + a + FNr

        dr = dt * Fr + dWr
        da = dt * Fa + dWa

        work = inv_work * (a * dr)
        girs = dt * (inv_gpos * FNr ** 2 + inv_gvel * FNa ** 2)
        step_loss = jnp.mean(girs - lmbda * work) / norm

        acc_next = acc + Measure(params, g, a) if measure else acc
        return (r + dr, a + da, acc_next, k), step_loss

    init = (r, a, jnp.zeros(4), key)
    (r_f, a_f, acc, _), step_losses = jax.lax.scan(step_fn, init, None, length=steps)

    acc = acc / steps
    err_f = jnp.sqrt(acc[:2] / jnp.maximum(acc[2:], 1e-30))
    return N * jnp.sum(step_losses), (r_f, a_f, err_f)

# ═════════════════════════════════════════════════════════════════════════
# JIT Compiled steps
# ═════════════════════════════════════════════════════════════════════════
@jax.jit
def update_step(params, opt_state, r, a, key):
    key, ks = jax.random.split(key)
    (loss, (r_f, a_f, _)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, r, a, ks, tstep, measure=False)
    updates, opt_state = Optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, r_f, a_f, key

@partial(jax.jit, static_argnames=('steps',))
def eval_measure_step(params, r, a, key, steps):
    key, ks = jax.random.split(key)
    loss, (r_f, a_f, err) = loss_fn(params, r, a, ks, steps, measure=True)
    return loss, r_f, a_f, err, key

@partial(jax.jit, static_argnames=('steps',))
def eval_step(params, r, a, key, steps):
    key, ks = jax.random.split(key)
    loss, (r_f, a_f, _) = loss_fn(params, r, a, ks, steps, measure=False)
    return loss, r_f, a_f, key

# ═════════════════════════════════════════════════════════════════════════
# Result list
# ═════════════════════════════════════════════════════════════════════════
LOSS  = []
SCGF  = []
ERROR = []

def dump():
    with open(f"{RESULT_DIR}/FCN_SCGF_{suffix}.pkl",  "wb") as f: pickle.dump(SCGF,  f)
    with open(f"{RESULT_DIR}/FCN_LOSS_{suffix}.pkl",  "wb") as f: pickle.dump(LOSS,  f)
    with open(f"{RESULT_DIR}/FCN_ERROR_{suffix}.pkl", "wb") as f: pickle.dump(ERROR, f)

# ═════════════════════════════════════════════════════════════════════════
# Warm-up
# ═════════════════════════════════════════════════════════════════════════
_, r, a, err, key = eval_measure_step(params, r, a, key, 5 * tstep)
ERROR.append(np.asarray(err))

# ═════════════════════════════════════════════════════════════════════════
# Train
# ═════════════════════════════════════════════════════════════════════════
for epoch in trange(epochs):
    params, opt_state, loss, r, a, key = update_step(params, opt_state, r, a, key)
    loss_val = float(loss)
    LOSS.append(loss_val)

    # Divergence check
    if not math.isfinite(loss_val):
        msg = (f"Diverged! Epoch: {epoch}, N: {N}, HID: {n_hid}, BLOCK: {n_blocks}, LMBDA: {lmbda}\n")
        with open(f"{DIVERGE_DIR}/FCN_{suffix}.txt", "w") as f: f.write(msg)
        raise RuntimeError("Diverge")

    if (epoch + 1) % mid_epochs == 0:
        _, _, _, err, key = eval_measure_step(params, r, a, key, 5 * tstep)
        ERROR.append(np.asarray(err))

    if epoch >= (epochs - n_models):
        loss_eval, _, _, key = eval_step(params, r, a, key, 20 * tstep)
        SCGF.append(float(loss_eval))

# ═════════════════════════════════════════════════════════════════════════
# Result
# ═════════════════════════════════════════════════════════════════════════
dump()