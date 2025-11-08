# stress_bench.py
# Pruebas de estrés sobre MinesweeperJAX: throughput sin gráficos.
# Ejecuta B tableros en paralelo durante T pasos dentro de un jit.
# Mide steps/s y celdas/s. Incluye políticas first_valid y random_valid.
# Tips: para reducir prealocación de memoria en GPU, ver variables XLA_* más abajo.

from __future__ import annotations
import os, time, math, argparse
import numpy as np

# === (Opcional) Ajustes de memoria GPU: deben setearse ANTES de importar jax ===
# Descomenta si querés evitar la prealocación de casi toda la VRAM.
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"    # asigna memoria a demanda
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"    # prealoca ~85% (si prealloc está activa)

import jax
import jax.numpy as jnp
from jax import lax

# Importa tu entorno
from env import MinesweeperJAX, MinesState  # asume env/__init__.py accesible

# ------------------------- Políticas (JAX-friendly) -------------------------
def policy_first_valid(env: MinesweeperJAX, s: MinesState, key: jax.Array | None) -> jax.Array:
    # (B, H*W) -> (B,)
    mask = env.action_mask(s).astype(jnp.int32)  # bool -> int
    # argmax elige la primera válida por fila
    return jnp.argmax(mask, axis=1).astype(jnp.int32)

def policy_random_valid(env: MinesweeperJAX, s: MinesState, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Muestrea acción al azar entre válidas con Gumbel-max.
    Retorna (actions, key2).
    """
    mask = env.action_mask(s)  # (B, H*W) bool
    B, A = mask.shape
    key, sub = jax.random.split(key)
    # Gumbel noise
    u = jax.random.uniform(sub, shape=(B, A), minval=1e-6, maxval=1.0 - 1e-6)
    g = -jnp.log(-jnp.log(u))
    # logits 0 en válidas, -inf en inválidas
    logits = jnp.where(mask, 0.0, -1e9)
    a = jnp.argmax(logits + g, axis=1).astype(jnp.int32)
    return a, key

# ----------------------- Rollouts (dos variantes de loop) --------------------
def make_rollout_scan(env: MinesweeperJAX, T: int, policy_name: str):
    """
    Rollout con lax.scan (compilación más rápida, HLO compacto).
    """
    if policy_name == "first_valid":
        def body(carry, t):
            s, key = carry
            a = policy_first_valid(env, s, key=None)
            s2, r, d = env.step(s, a)
            return (s2, key), (r, d)
    elif policy_name == "random_valid":
        def body(carry, t):
            s, key = carry
            a, key2 = policy_random_valid(env, s, key)
            s2, r, d = env.step(s, a)
            return (s2, key2), (r, d)
    else:
        raise ValueError("policy_name inválido")

    def rollout(state: MinesState, key: jax.Array):
        (sT, kT), (R, D) = lax.scan(body, (state, key), jnp.arange(T))
        return sT, R, D
    return rollout

def make_rollout_unrolled(env: MinesweeperJAX, T: int, policy_name: str):
    """
    Rollout con bucle Python dentro del jit (unrolled).
    Suele dar HLO enorme (compilaciones muy pesadas) pero en GPU a veces rinde más.
    """
    def rollout(state: MinesState, key: jax.Array):
        s, k = state, key
        R_list, D_list = [], []
        for _ in range(T):  # se desenrolla al trazar
            if policy_name == "first_valid":
                a = policy_first_valid(env, s, None)
            else:
                a, k = policy_random_valid(env, s, k)
            s, r, d = env.step(s, a)
            R_list.append(r); D_list.append(d)
        R = jnp.stack(R_list, axis=0)  # (T, B)
        D = jnp.stack(D_list, axis=0)
        return s, R, D
    return rollout

# ----------------------------- Benchmark runner ------------------------------
def bench(env: MinesweeperJAX, B: int, T: int, mode: str, policy: str, warmup_T: int = 4):
    key = jax.random.key(0)

    # Reset (se puede jittear con B estático si querés)
    state = env.reset(key, batch_size=B)

    if mode == "scan":
        rollout = make_rollout_scan(env, T, policy)
    elif mode == "unrolled":
        rollout = make_rollout_unrolled(env, T, policy)
    else:
        raise ValueError("mode inválido")

    # JIT: T queda capturado en el closure; shapes dependen de B y H,W
    rollout_jit = jax.jit(rollout)

    # Warmup (compilación) + sincronización
    s_w, R_w, D_w = rollout_jit(state, key)
    _ = jax.device_get((s_w.revealed, R_w, D_w))  # sync total

    # Medición
    t0 = time.perf_counter()
    s_f, R_f, D_f = rollout_jit(state, key)
    # bloquea hasta terminar (muy importante para medir realmente)
    _ = jax.device_get((s_f.revealed, R_f, D_f))
    dt = time.perf_counter() - t0

    steps = B * T
    steps_per_s = steps / dt
    cells_per_s = steps_per_s * (env.H * env.W)

    return {
        "B": B, "T": T, "mode": mode, "policy": policy,
        "seconds": dt,
        "steps": int(steps),
        "steps_per_s": steps_per_s,
        "cells_per_s": cells_per_s,
    }

# ------------------------------- CLI & main -----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--W", type=int, default=16)
    parser.add_argument("--n_mines", type=int, default=40)   # número de minas en el tablero
    parser.add_argument("--R", type=int, default=1)          # context_radius
    parser.add_argument("--T", type=int, default=128)        # pasos por benchmark
    parser.add_argument("--mode", choices=["scan", "unrolled"], default="scan")
    parser.add_argument("--policy", choices=["first_valid", "random_valid"], default="first_valid")
    parser.add_argument("--batches", type=str, default="256,512,1024,2048,4096")
    args = parser.parse_args()

    print(f"Device(s): {[d.platform + ':' + d.device_kind for d in jax.devices()]}")
    env = MinesweeperJAX(H=args.H, W=args.W, n_mines=args.n_mines, context_radius=args.R)

    batch_list = [int(x) for x in args.batches.split(",") if x.strip()]
    print(f"Running: H={args.H} W={args.W} R={args.R} T={args.T} mode={args.mode} policy={args.policy}")
    print("Batches:", batch_list)
    print("-" * 80)

    rows = []
    for B in batch_list:
        try:
            res = bench(env, B=B, T=args.T, mode=args.mode, policy=args.policy)
            rows.append(res)
            print(f"B={B:5d}  T={args.T:4d}  {args.mode:8s}  {args.policy:12s}  "
                  f"steps/s={res['steps_per_s']:.1f}  cells/s={res['cells_per_s']:.1f}  time={res['seconds']:.3f}s")
        except Exception as e:
            print(f"B={B}: ERROR: {e}")

    # resumen
    if rows:
        best = max(rows, key=lambda r: r["steps_per_s"])
        print("-" * 80)
        print("MEJOR:", f"B={best['B']}", f"steps/s={best['steps_per_s']:.1f}", f"cells/s={best['cells_per_s']:.1f}")

if __name__ == "__main__":
    main()
