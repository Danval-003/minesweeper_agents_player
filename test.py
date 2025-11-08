from minesweeper_env_rl import MinesweeperJAX, build_jit_env
import jax
import jax.numpy as jnp

# --------------------------------- Demo ---------------------------------------
if __name__ == "__main__":
    # Comprobar que se tiene GPU sin romper si JAX no incluye backend CUDA
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []

    if gpu_devices:
        print("GPU disponible")
    else:
        print("No se encontró GPU, utilizando CPU")
        

    key = jax.random.PRNGKey(0)
    env = MinesweeperJAX(H=16, W=16, mine_prob=0.15625, context_radius=1)
    reset_jit, observe_jit, mask_jit, step_jit = build_jit_env(env)

    B = 2048  # subí esto si tenés GPU
    state = reset_jit(key, B)

    obs = observe_jit(state)          # (B, C, H, W)
    mask = mask_jit(state)            # (B, H*W)

    # Política tonta: tomar la primera acción válida
    first_valid = jnp.argmax(mask, axis=1)

    state, r, d = step_jit(state, first_valid)
    print("obs", obs.shape, "reward mean", r.mean(), "done%", d.mean())
