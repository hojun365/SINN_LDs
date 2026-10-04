# ═════════════════════════════════════════════════════════════════════════
# Library
# ═════════════════════════════════════════════════════════════════════════
import os
import math
import pickle
import argparse
from functools import partial
import jax
import jax.numpy as jnp
import optax
from tqdm import trange
import Model as MM

# ═════════════════════════════════════════════════════════════════════════
# Parser
# ═════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--hid", type=int, required=True)
parser.add_argument("--block", type=int, required=True)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--lmbda", type=float, required=True)
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--tstep", type=int, default=5000)
args = parser.parse_args()

N = args.N
lr = args.lr
n_hid = args.hid
lmbda = args.lmbda
epochs = args.epoch
n_blocks = args.block

# ═════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════
RESULT_DIR = "./result"
DIVERGE_DIR = "./Diverge"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DIVERGE_DIR, exist_ok=True)
suffix = f"N={N}_H={n_hid}_B={n_blocks}_λ={lmbda}_tstep={args.tstep}"

# ═════════════════════════════════════════════════════════════════════════
# Physical & numerical parameters
# ═════════════════════════════════════════════════════════════════════════
B        = 200
n_models = 100
tstep    = args.tstep
dt       = args.dt
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
# Network and optimizer
# ═════════════════════════════════════════════════════════════════════════
NN = MM.EQI(N=N, n_hid=n_hid, n_block=n_blocks)
params = NN.init(key_model, r, a, lmbda)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.scale(-lr))
opt_state = optimizer.init(params)

# ═════════════════════════════════════════════════════════════════════════
# Loss
# ═════════════════════════════════════════════════════════════════════════
def loss_fn(params, r, a, key, steps):
    norm = dt * steps
    Ws = NN.apply(params, method=NN.build_Ws)

    @partial(jax.checkpoint, prevent_cse=False)
    def step_fn(carry, _):
        r, a, key = carry
        key, key_r, key_a = jax.random.split(key, 3)

        dWr = Vr * jax.random.normal(key_r, r.shape, dtype=r.dtype)
        dWa = Va * jax.random.normal(key_a, a.shape, dtype=a.dtype)

        g = (jnp.roll(r, -1, axis=-1) -r + GAP) * N
        FNr, FNa = NN.apply(params, g, a, lmbda, Ws)

        Fa = -a / τ + FNa
        Fr = μ * K * (jnp.roll(r, -1, axis=-1) - 2.0 * r + jnp.roll(r, 1, axis=-1) + PBC) + a + FNr

        dr = dt * Fr + dWr
        da = dt * Fa + dWa

        work = inv_work * a * dr
        girs = dt * (inv_gpos * FNr**2 + inv_gvel * FNa**2)
        step_loss = jnp.mean(girs - lmbda * work) / norm

        return (r + dr, a + da, key), step_loss

    (r_final, a_final, _), step_losses = jax.lax.scan(step_fn, (r, a, key), xs=None, length=steps)

    loss = N * jnp.sum(step_losses)
    return loss, (r_final, a_final)

# ═════════════════════════════════════════════════════════════════════════
# JIT
# ═════════════════════════════════════════════════════════════════════════
@jax.jit
def update_step(params, opt_state, r, a, key):
    key, simulation_key = jax.random.split(key)
    (loss, (r_final, a_final)), grads = jax.value_and_grad(loss_fn, has_aux=True,)(params, r, a, simulation_key, tstep)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return (params, opt_state, loss, r_final, a_final, key)


@partial(jax.jit, static_argnames=("steps",))
def eval_step(params, r, a, key, steps):
    key, simulation_key = jax.random.split(key)
    loss, (r_final, a_final) = loss_fn(params, r, a, simulation_key, steps)

    return (loss, r_final, a_final, key)

# ═════════════════════════════════════════════════════════════════════════
# Warm-up
# ═════════════════════════════════════════════════════════════════════════
LOSS = []
SCGF = []
_, r, a, key = eval_step(params, r, a, key, tstep)

# ═════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════
for epoch in trange(epochs):
    params, opt_state, loss, r, a, key = update_step(params, opt_state, r, a, key)
    loss_value = float(loss)
    LOSS.append(loss_value)

    if not math.isfinite(loss_value):
        raise RuntimeError("Diverged")

    if epoch >= (epochs - n_models):
        loss_eval, _, _, key = eval_step(params, r, a, key, 20*tstep)
        SCGF.append(float(loss_eval))


# ═════════════════════════════════════════════════════════════════════════
# Results
# ═════════════════════════════════════════════════════════════════════════
scgf_file = f"{RESULT_DIR}/EQI_SCGF_{suffix}.pkl"
loss_file = f"{RESULT_DIR}/EQI_LOSS_{suffix}.pkl"

with open(scgf_file, "wb") as file:
    pickle.dump(SCGF, file)

with open(loss_file, "wb") as file:
    pickle.dump(LOSS, file)