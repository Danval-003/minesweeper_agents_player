"""
Minesweeper RL env (JAX, GPU-ready, batched, jit/vmap friendly)
---------------------------------------------------------------

• Observación: canales apilados (one-hot 0..8, máscara de no revelado, máscara de banderas).
• Acción: índice plano [0, H*W). Se enmascaran inválidas (~revealed | flagged).
• Lógica de revelar: incluye flood-fill de ceros con dilatación 8-vecinos y borde numérico.
• Reset y step vectorizados (primer dimensión = batch). Pensado para jit y miles de tableros.

Requisitos: jax>=0.4, jaxlib con CUDA si vas a GPU.

API principal:
    env = MinesweeperJAX(H=16, W=16, mine_prob=0.15625, dtype=jnp.float32)
    state = env.reset(key, batch_size=1024)  # GPU
    obs = env.observe(state)
    mask = env.action_mask(state)
    state, reward, done = env.step(state, actions)

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

from jax.tree_util import register_pytree_node

from functools import partial

Array = jnp.ndarray


def _make_kernel_3x3_eight_neighbors() -> Array:
    k = jnp.ones((3, 3), dtype=jnp.int32)
    k = k.at[1, 1].set(0)
    return k


def _conv2d_ones(x: Array) -> Array:
    """Suma en vecindad 3×3 (8 vecinos) con ReduceWindow; NO cuDNN.
    x: (B,H,W) -> (B,H,W), dtype entero.
    """
    x_i32 = x.astype(jnp.int32)
    s = lax.reduce_window(
        x_i32,
        jnp.array(0, jnp.int32),
        lax.add,
        window_dimensions=(1, 3, 3),
        window_strides=(1, 1, 1),
        padding=((0, 0), (1, 1), (1, 1))  # SAME
    )
    # quitar el centro para contar solo 8 vecinos
    return s - x_i32

@dataclass
class MinesState:
    mines: Array        # (B,H,W) bool
    revealed: Array     # (B,H,W) bool
    flagged: Array      # (B,H,W) bool
    numbers: Array      # (B,H,W) uint8 (0..8)
    done: Array         # (B,) bool
    rng_key: Array      # PRNGKey


class MinesweeperJAX:
    def __init__(self, H: int = 16, W: int = 16, mine_prob: float = 0.15625,
                 dtype=jnp.float32, reward_safe: float = 0.1, reward_boom: float = -1.0,
                 reward_win: float = 1.0,
                 context_radius: int = 0):
        self.H, self.W = H, W
        self.N = H * W
        self.mine_prob = mine_prob
        self.dtype = dtype
        self.reward_safe = dtype(reward_safe)
        self.reward_boom = dtype(reward_boom)
        self.reward_win = dtype(reward_win)
        self.context_radius = int(context_radius)

    # ----------------------------- Reset ---------------------------------
    def _place_mines(self, key: Array, B: int) -> Tuple[Array, Array]:
        """Bernoulli por celda. Devuelve (key, mines[Bool])."""
        key, sub = jax.random.split(key)
        mines = jax.random.bernoulli(sub, p=self.mine_prob, shape=(B, self.H, self.W))
        # Evitar tablero vacío o lleno (opcional):
        def _fix(m):
            total = m.sum()
            m = lax.cond(total == 0, lambda _: m.at[0, 0].set(True), lambda _: m, None)
            m = lax.cond(total == (self.H * self.W), lambda _: m.at[0, 0].set(False), lambda _: m, None)
            return m
        mines = jax.vmap(_fix)(mines)
        return key, mines

    def _compute_numbers(self, mines: Array) -> Array:
        # Conteo de minas vecinas por suma de conv; mines: bool -> int32 -> uint8
        neigh = _conv2d_ones(mines.astype(jnp.int32))
        # Donde hay mina, número suele ser irrelevante, lo dejamos 0
        return neigh.astype(jnp.uint8)

    def reset(self, key: Array, batch_size: int = 1) -> MinesState:
        key, mines = self._place_mines(key, batch_size)
        numbers = self._compute_numbers(mines)
        revealed = jnp.zeros_like(mines, dtype=bool)
        flagged = jnp.zeros_like(mines, dtype=bool)
        done = jnp.zeros((batch_size,), dtype=bool)
        return MinesState(mines=mines, revealed=revealed, flagged=flagged,
                          numbers=numbers, done=done, rng_key=key)

    # --------------------------- Observación ------------------------------
    def observe(self, s: MinesState) -> Array:
        r"""Devuelve (B, C, H, W) con:
        • 9 canales one-hot(0..8) en celdas reveladas (0 en no reveladas)
        • 1 canal mask no revelado
        • 1 canal mask flagged
        • + contexto local explícito si context_radius>0:
            – Para cada (dx,dy) en el vecindario (2R+1)^2 {0,0}, 1 canal con el número
              de la celda vecina (si está revelada; si no, 0). Canales "shifted" sin wrap-around.
            – 2 canales de resumen: conteo de no reveladas y de banderas en la ventana.
        """
        B, H, W = s.revealed.shape
        # one-hot 0..8 (sólo donde revelado)
        onehot = jax.nn.one_hot(s.numbers, 9, dtype=self.dtype)  # (B,H,W,9)
        onehot = jnp.moveaxis(onehot, -1, 1)                     # (B,9,H,W)
        onehot = onehot * s.revealed[:, None, :, :].astype(self.dtype)
        mask_unrev = (~s.revealed).astype(self.dtype)[:, None, :, :]
        mask_flag = s.flagged.astype(self.dtype)[:, None, :, :]

        feats = [onehot, mask_unrev, mask_flag]

        R = self.context_radius
        if R > 0:
            # mapa de números visibles (0 en no reveladas)
            num_visible = s.numbers.astype(self.dtype) * s.revealed.astype(self.dtype)  # (B,H,W)
            ctx_maps = []
            for dx in range(-R, R+1):
                for dy in range(-R, R+1):
                    if dx == 0 and dy == 0:
                        continue
                    pad_x = ((0, 0), (max(dx, 0), max(-dx, 0)), (max(dy, 0), max(-dy, 0)))
                    nv_pad = jnp.pad(num_visible, pad_x, mode="constant")
                    nv_shift = nv_pad[:, max(-dx, 0):max(-dx, 0) + H, max(-dy, 0):max(-dy, 0) + W]
                    ctx_maps.append(nv_shift[:, None, :, :])
            if ctx_maps:
                feats.append(jnp.concatenate(ctx_maps, axis=1))

            # Resúmenes en ventana (2R+1)x(2R+1) por conv SAME
            ksize = 2 * R + 1
            if ksize > 1:
                k = jnp.ones((ksize, ksize), dtype=self.dtype)

                def conv_same(x4):
                    return lax.conv_general_dilated(
                        lhs=x4,
                        rhs=k.reshape(1, 1, ksize, ksize),
                        window_strides=(1, 1), padding="SAME",
                        dimension_numbers=("NCHW", "OIHW", "NCHW"))

                unrev = (~s.revealed).astype(self.dtype)[:, None, :, :]
                flag = s.flagged.astype(self.dtype)[:, None, :, :]
                cnt_unrev = conv_same(unrev)
                cnt_flag = conv_same(flag)
                # quitar el centro
                cnt_unrev = cnt_unrev - unrev
                cnt_flag = cnt_flag - flag
                feats.extend([cnt_unrev, cnt_flag])

        return jnp.concatenate(feats, axis=1)

    def action_mask(self, s: MinesState) -> Array:
        """(B, H*W) bool: True = acción válida (no revelado y no bandera y no done)."""
        valid = (~s.revealed) & (~s.flagged)
        valid = valid.reshape(valid.shape[0], -1)
        # Si ya está done, ninguna válida
        valid = jnp.where(s.done[:, None], jnp.zeros_like(valid), valid)
        return valid

    # ------------------------------ Step ---------------------------------
    def _reveal_cells(self, s: MinesState, actions: Array) -> Tuple[Array, Array]:
        """Aplica la lógica de revelar: flood-fill de ceros + borde numérico.
        Retorna (new_revealed[bool], hit_mine[bool]) por batch.
        """
        B, H, W = s.revealed.shape
        idx_r = actions // self.W
        idx_c = actions % self.W
        sel = jnp.zeros((B, H, W), dtype=bool)
        sel = sel.at[jnp.arange(B), idx_r, idx_c].set(True)

        # Si ya estaba revelada o marcada, no hacemos nada (se gestiona en step)
        sel_valid = sel & (~s.revealed) & (~s.flagged)

        # Mina golpeada (por batch)
        hit_mine = (sel_valid & s.mines).any(axis=(1, 2))

        # Rama segura: abrir ceros conectados + borde numérico
        def open_safe_branch(_):
            numbers = s.numbers
            zeros = (numbers == 0)
            start_zero = sel_valid & zeros
            start_num = sel_valid & (~zeros) & (~s.mines)

            def body(carry):
                opened, frontier = carry
                opened = opened | frontier
                # Expandimos a 8-vecinos dentro de zeros y no abiertos
                neigh = _conv2d_ones(frontier.astype(jnp.int32)) > 0
                next_frontier = neigh & zeros & (~opened)
                return (opened, next_frontier)

            def cond(carry):
                _, frontier = carry
                return frontier.any()

            opened0 = jnp.zeros_like(s.revealed)
            opened_zeros, _ = lax.while_loop(cond, body, (opened0, start_zero))

            # Bordes numéricos adyacentes a cualquier cero abierto
            neigh_open_zero = (_conv2d_ones(opened_zeros.astype(jnp.int32)) > 0)
            border_nums = neigh_open_zero & (~zeros) & (~s.mines)

            # Celdas numéricas iniciales (click directo en número)
            opened_nums_click = start_num

            new_revealed = s.revealed | opened_zeros | border_nums | opened_nums_click
            return new_revealed

        safe_revealed = open_safe_branch(None)
        # Selección por batch (broadcast correcto con jnp.where)
        new_revealed = jnp.where(hit_mine[:, None, None], s.revealed, safe_revealed)

        return new_revealed, hit_mine

    def _win_condition(self, revealed: Array, mines: Array) -> Array:
        # Gana si todas las NO-minas están reveladas
        return ((~mines) & (~revealed)).sum(axis=(1, 2)) == 0

    def step(self, s: MinesState, actions: Array) -> Tuple[MinesState, Array, Array]:
        """actions: (B,) int32 (plano). Retorna (new_state, reward[B], done[B])."""
        B = s.revealed.shape[0]
        actions = actions.astype(jnp.int32)

        # Acciones inválidas o episodios ya terminados → reward 0 (NOP)
        valid_mask = self.action_mask(s)
        a_valid = jnp.take_along_axis(valid_mask, actions[:, None], axis=1).squeeze(1)
        a_valid = a_valid & (~s.done)

        def do_nop(_):
            return s.revealed, jnp.zeros((B,), self.dtype)

        def do_reveal(_):
            new_rev, hit_mine = self._reveal_cells(s, actions)
            # Recompensas: +r_safe * (#nuevas), -1 si mina, +win si gana
            newly = (new_rev & (~s.revealed) & (~s.mines)).sum(axis=(1, 2)).astype(self.dtype)
            r = newly * self.reward_safe
            r = r + self.reward_boom * hit_mine.astype(self.dtype)
            win = self._win_condition(new_rev, s.mines)
            r = r + self.reward_win * win.astype(self.dtype)
            return new_rev, r

        new_revealed, r_step = lax.cond(a_valid.any(), do_reveal, do_nop, operand=None)

        # Recalcular condiciones terminales por batch
        idx_r = actions // self.W
        idx_c = actions % self.W
        pick = jnp.zeros_like(s.mines)
        pick = pick.at[jnp.arange(B), idx_r, idx_c].set(True)
        hit = (pick & (~s.revealed) & (~s.flagged) & s.mines).any(axis=(1, 2))
        win = self._win_condition(new_revealed, s.mines)

        done = s.done | hit | win

        new_s = MinesState(
            mines=s.mines,
            revealed=new_revealed,
            flagged=s.flagged,
            numbers=s.numbers,
            done=done,
            rng_key=s.rng_key,
        )
        return new_s, r_step, done


# ------------------------------- JIT wrappers ---------------------------------
# Opcional: compilar helpers para máximo rendimiento en loops de entrenamiento.

def build_jit_env(env):
    @partial(jax.jit, static_argnames=('B',))
    def reset_jit(key, B: int):
        return env.reset(key, B)

    observe_jit = jax.jit(env.observe)
    action_mask_jit = jax.jit(env.action_mask)
    step_jit = jax.jit(env.step)  # especializa por shape (B,)
    return reset_jit, observe_jit, action_mask_jit, step_jit


# --- Registrar MinesState como pytree ---
def _ms_flatten(ms: MinesState):
    # children: SOLO arrays/pytrees válidos para JAX
    children = (ms.mines, ms.revealed, ms.flagged, ms.numbers, ms.done, ms.rng_key)
    aux = None
    return children, aux


def _ms_unflatten(aux, children):
    mines, revealed, flagged, numbers, done, rng_key = children
    return MinesState(mines=mines, revealed=revealed, flagged=flagged,
                      numbers=numbers, done=done, rng_key=rng_key)


# ------------------------------- RENDER: RGB -----------------------------------
import numpy as _np
import matplotlib.pyplot as _plt

def render_rgb(state: MinesState, i: int = 0, scale: int = 24, reveal_mines_on_done: bool = True) -> _np.ndarray:
    """
    Devuelve un frame RGB (H*scale, W*scale, 3) uint8 del tablero i.
    Colores:
      · No revelada: gris oscuro
      · Bandera: amarillo
      · Revelada: paleta por número (0..8)
      · Si el episodio terminó y reveal_mines_on_done=True: minas en rojo
    """
    mines   = _np.asarray(jax.device_get(state.mines[i]))
    revel   = _np.asarray(jax.device_get(state.revealed[i]))
    flags   = _np.asarray(jax.device_get(state.flagged[i]))
    nums    = _np.asarray(jax.device_get(state.numbers[i]))
    done    = bool(_np.asarray(jax.device_get(state.done[i])))

    H, W = revel.shape
    img = _np.zeros((H, W, 3), dtype=_np.uint8)

    # Paleta simple para números 0..8
    palette = _np.array([
        [200, 200, 200],  # 0
        [ 48, 130, 255],  # 1
        [ 64, 160,  64],  # 2
        [220,  60,  60],  # 3
        [ 64,  64, 160],  # 4
        [160,  64,  64],  # 5
        [ 64, 160, 160],  # 6
        [  0,   0,   0],  # 7
        [128, 128, 128],  # 8
    ], dtype=_np.uint8)

    # Base: no reveladas = gris oscuro
    img[:] = _np.array([50, 50, 50], dtype=_np.uint8)

    # Banderas
    img[flags] = _np.array([240, 220, 70], dtype=_np.uint8)

    # Reveladas: colorear por número
    rmask = revel & (~flags)
    img[rmask] = palette[nums[rmask]]

    # Si terminó y queremos ver minas: píntalas en rojo
    if done and reveal_mines_on_done:
        img[mines] = _np.array([255, 0, 0], dtype=_np.uint8)

    # Escalar para que se vea grande
    if scale > 1:
        img = _np.repeat(_np.repeat(img, scale, axis=0), scale, axis=1)
    return img

def show_once(state: MinesState, i: int = 0, scale: int = 24):
    _plt.figure(figsize=(6, 6))
    _plt.imshow(render_rgb(state, i=i, scale=scale))
    _plt.axis("off")
    _plt.tight_layout()
    _plt.show()


def view_episode(env: MinesweeperJAX, state: MinesState, steps: int = 50, policy="first_valid", i: int = 0):
    _plt.ion()
    fig = _plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(render_rgb(state, i=i, scale=28))
    ax.axis("off")
    fig.canvas.draw(); fig.canvas.flush_events()

    for t in range(steps):
        # política simple
        mask = _np.asarray(jax.device_get(env.action_mask(state)))
        if policy == "first_valid":
            a = mask.argmax(axis=1)          # (B,)
        else:
            # random entre válidas
            valid_idx = _np.where(mask[0])[0]
            if len(valid_idx) == 0:
                break
            a = _np.zeros((mask.shape[0],), dtype=_np.int32)
            a[0] = _np.random.choice(valid_idx)

        # paso
        state, r, d = env.step(state, jnp.asarray(a))
        frame = render_rgb(state, i=i, scale=28)
        im.set_data(frame)
        fig.canvas.draw(); fig.canvas.flush_events()
        if bool(_np.asarray(jax.device_get(d[i]))):
            break
    _plt.ioff()
    _plt.show()

register_pytree_node(MinesState, _ms_flatten, _ms_unflatten)

# --------------------------------- Demo ---------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    env = MinesweeperJAX(H=16, W=16, mine_prob=0.15625, context_radius=1)
    reset_jit, observe_jit, mask_jit, step_jit = build_jit_env(env)

    B = 2048  # subí esto si tenés GPU
    state = reset_jit(key, B)
    print("Initial state:", state)

    obs = observe_jit(state)          # (B, C, H, W)
    mask = mask_jit(state)            # (B, H*W)

    # Política tonta: tomar la primera acción válida
    first_valid = jnp.argmax(mask, axis=1)

    state, r, d = step_jit(state, first_valid)
    print("obs", obs.shape, "reward mean", r.mean(), "done%", d.mean())
