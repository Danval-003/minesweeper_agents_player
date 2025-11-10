# stress_bench.py
# Pruebas de estrés sobre MinesweeperPOMDP: throughput sin gráficos.
# Crea B entornos en paralelo (a nivel Python), hace T pasos por cada uno,
# usando políticas "first_valid" o "random_valid".
# Mide steps/s y celdas/s.

from __future__ import annotations
import time
import argparse
import random
from typing import Dict, List

import numpy as np

from env import MinesweeperPOMDP  # asume que este archivo está en env.py o similar


# ------------------------- Políticas sobre obs['valid_actions'] -------------------------

def policy_first_valid(obs: Dict) -> Dict:
    """
    Elige la primera acción de tipo 'reveal' en obs['valid_actions'].
    Si no hay ninguna, elige la primera acción que exista.
    Retorna un dict acción: {'cell': (r,c), 'type': 'reveal' | 'flag' | 'unflag'}.
    """
    valid_actions: List[Dict] = obs.get("valid_actions", [])
    if not valid_actions:
        return {"cell": (0, 0), "type": "reveal"}  # fallback

    # Priorizar reveals
    for a in valid_actions:
        if a["type"] == "reveal":
            return a

    # Si no hay reveals (raro), tomar la primera acción cualquiera
    return valid_actions[0]


def policy_random_valid(obs: Dict, rng: random.Random) -> Dict:
    """
    Elige una acción aleatoria de tipo 'reveal'.
    Si no hay reveals, elige cualquier acción aleatoria disponible.
    """
    valid_actions: List[Dict] = obs.get("valid_actions", [])
    if not valid_actions:
        return {"cell": (0, 0), "type": "reveal"}  # fallback

    reveal_actions = [a for a in valid_actions if a["type"] == "reveal"]
    if reveal_actions:
        return rng.choice(reveal_actions)
    return rng.choice(valid_actions)


# ------------------------- Loop de un entorno individual -------------------------

def run_env_steps(env: MinesweeperPOMDP,
                  T: int,
                  policy: str,
                  rng: random.Random) -> int:
    """
    Ejecuta T llamadas a env.step(...) sobre UN entorno.
    Cada vez que el episodio termina, se hace reset y se sigue.
    Devuelve el número total de steps ejecutados (debería ser T).
    """
    obs = env.reset()  # primer reset
    steps = 0

    for _ in range(T):
        valid_actions = obs.get("valid_actions", [])
        if not valid_actions:
            # Si por alguna razón no hay acciones válidas, reset y continuar
            obs = env.reset()
            continue

        if policy == "first_valid":
            action = policy_first_valid(obs)
        elif policy == "random_valid":
            action = policy_random_valid(obs, rng)
        else:
            raise ValueError(f"Política inválida: {policy}")

        obs, reward, done, info = env.step(action)
        steps += 1

        if done:
            # Reiniciar el episodio para seguir cargando la CPU
            obs = env.reset()

    return steps


# ----------------------------- Benchmark runner --------------------------------

def bench(rows: int,
          cols: int,
          num_mines: int,
          window_radius: int,
          window_shape: str,
          B: int,
          T: int,
          policy: str,
          seed: int = 0):
    """
    Crea B entornos MinesweeperPOMDP y ejecuta T pasos por cada uno.
    Retorna métricas de rendimiento.
    """
    rng = random.Random(seed)

    # Crear todos los entornos
    envs = [
        MinesweeperPOMDP(
            rows=rows,
            cols=cols,
            num_mines=num_mines,
            window_radius=window_radius,
            window_shape=window_shape,
        )
        for _ in range(B)
    ]

    # Warmup corto (opcional, para que no contamine el tiempo con la primera llamada)
    for env in envs[:1]:
        _ = run_env_steps(env, T=8, policy=policy, rng=rng)

    # Medición real
    t0 = time.perf_counter()
    total_steps = 0
    for env in envs:
        total_steps += run_env_steps(env, T=T, policy=policy, rng=rng)
    dt = time.perf_counter() - t0

    steps_per_s = total_steps / dt if dt > 0 else float("inf")
    cells_per_s = steps_per_s * (rows * cols)

    return {
        "rows": rows,
        "cols": cols,
        "num_mines": num_mines,
        "window_radius": window_radius,
        "window_shape": window_shape,
        "B": B,
        "T": T,
        "policy": policy,
        "seconds": dt,
        "steps": total_steps,
        "steps_per_s": steps_per_s,
        "cells_per_s": cells_per_s,
    }


# ------------------------------- CLI & main -----------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=16, help="Número de filas")
    parser.add_argument("--W", type=int, default=16, help="Número de columnas")
    parser.add_argument("--n_mines", type=int, default=40, help="Número de minas")
    parser.add_argument("--R", type=int, default=2, help="Radio de ventana de observación")
    parser.add_argument("--window_shape",
                        type=str,
                        default="rhombus",
                        choices=["rhombus", "circle", "square"])
    parser.add_argument("--T", type=int, default=128,
                        help="Pasos por entorno en el benchmark")
    parser.add_argument("--policy",
                        choices=["first_valid", "random_valid"],
                        default="first_valid")
    parser.add_argument("--batches", type=str,
                        default="8,16,32,64,128",
                        help="Lista de tamaños B separados por coma")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = args.H
    cols = args.W

    print(f"Running MinesweeperPOMDP stress bench")
    print(f"Grid: {rows}x{cols} | mines={args.n_mines} | window R={args.R} shape={args.window_shape}")
    print(f"T per env: {args.T} | policy={args.policy}")
    batch_list = [int(x) for x in args.batches.split(",") if x.strip()]
    print("Batches:", batch_list)
    print("-" * 80)

    results = []
    for B in batch_list:
        try:
            res = bench(
                rows=rows,
                cols=cols,
                num_mines=args.n_mines,
                window_radius=args.R,
                window_shape=args.window_shape,
                B=B,
                T=args.T,
                policy=args.policy,
                seed=args.seed,
            )
            results.append(res)
            print(
                f"B={B:5d}  T={args.T:4d}  policy={args.policy:12s}  "
                f"steps={res['steps']:7d}  "
                f"steps/s={res['steps_per_s']:10.1f}  "
                f"cells/s={res['cells_per_s']:10.1f}  "
                f"time={res['seconds']:.3f}s"
            )
        except Exception as e:
            print(f"B={B}: ERROR: {e}")

    if results:
        best = max(results, key=lambda r: r["steps_per_s"])
        print("-" * 80)
        print("MEJOR:",
              f"B={best['B']}",
              f"steps/s={best['steps_per_s']:.1f}",
              f"cells/s={best['cells_per_s']:.1f}")


if __name__ == "__main__":
    main()
