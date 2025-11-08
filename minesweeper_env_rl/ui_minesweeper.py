# ui_minesweeper.py
# Interfaz interactiva para el entorno MinesweeperJAX:
# - Click sobre el tablero para jugar de forma manual.
# - Botones para "paso del agente" (first_valid / random).
# - Panel "lo que ve el agente": radio R, ventana local (2R+1)x(2R+1),
#   y mapas de resumen (conteo de no-reveladas, banderas) calculados por env.observe.
# Requiere: streamlit, streamlit-image-coordinates, pillow, imageio, jax, jaxlib.

from __future__ import annotations
import os
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw, ImageFont

import jax
import jax.numpy as jnp

# Importa tu entorno
from env import MinesweeperJAX, MinesState

# ------------------------------ Paleta / estilos ------------------------------
_MS_BG_HIDDEN   = (50, 50, 50)
_MS_BG_REVEALED = (215, 215, 215)
_MS_BG_ZERO     = (230, 230, 230)
_MS_FLAG        = (240, 220, 70)
_MS_GRID        = (25, 25, 25)
_MS_BOMB        = (200, 20, 20)
_MS_HILITE      = (255, 160, 0)
_MS_MASK_VALID  = (70, 160, 70)  # overlay verdoso para acciones válidas

_MS_NUM_COLORS = {
    1: ( 48,130,255),
    2: ( 64,160, 64),
    3: (220, 60, 60),
    4: ( 64, 64,160),
    5: (160, 64, 64),
    6: ( 64,160,160),
    7: (  0,  0,  0),
    8: (128,128,128),
}

def _safe_font(px: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", px)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", px)
        except Exception:
            return ImageFont.load_default()

def _measure_text(draw: ImageDraw.ImageDraw, txt: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    # Pillow >=10: textbbox; fallback a font.getbbox
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), txt, font=font)
        return (right - left), (bottom - top)
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(txt)
        return (right - left), (bottom - top)
    # último recurso
    return font.getsize(txt)

def _index_to_rc(idx: int, W: int):
    return int(idx // W), int(idx % W)

def _rc_to_index(r: int, c: int, W: int):
    return r * W + c

# ------------------------------ Render tablero ------------------------------
def render_board_img(state: MinesState,
                     i: int = 0,
                     cell_px: int = 28,
                     highlight_rc: tuple[int,int] | None = None,
                     show_valid_mask: bool = False,
                     env: MinesweeperJAX | None = None) -> Image.Image:
    """Devuelve un PIL.Image del tablero i (B=1 usualmente)."""
    mines    = np.asarray(jax.device_get(state.mines[i]))
    revealed = np.asarray(jax.device_get(state.revealed[i]))
    flagged  = np.asarray(jax.device_get(state.flagged[i]))
    numbers  = np.asarray(jax.device_get(state.numbers[i]))

    H, W = revealed.shape
    img = Image.new("RGB", (W*cell_px, H*cell_px), _MS_BG_HIDDEN)
    draw = ImageDraw.Draw(img)
    font = _safe_font(max(12, int(cell_px*0.7)))

    # fondo de celdas
    for r in range(H):
        for c in range(W):
            x0, y0 = c*cell_px, r*cell_px
            x1, y1 = x0+cell_px-1, y0+cell_px-1

            if flagged[r, c] and not revealed[r, c]:
                draw.rectangle([x0+1, y0+1, x1-1, y1-1], fill=_MS_FLAG)
                draw.polygon([(x0+cell_px*0.3, y0+cell_px*0.75),
                              (x0+cell_px*0.3, y0+cell_px*0.25),
                              (x0+cell_px*0.7, y0+cell_px*0.45)], fill=(200,40,40))
                continue

            if revealed[r, c]:
                bg = _MS_BG_REVEALED if numbers[r, c] > 0 else _MS_BG_ZERO
                draw.rectangle([x0+1, y0+1, x1-1, y1-1], fill=bg)
                n = int(numbers[r, c])
                if n > 0:
                    txt = str(n)
                    tw, th = _measure_text(draw, txt, font)
                    cx, cy = x0 + (cell_px - tw)//2, y0 + (cell_px - th)//2 - 1
                    draw.text((cx, cy), txt, fill=_MS_NUM_COLORS.get(n, (0,0,0)), font=font)

    # grid
    for r in range(H+1):
        y = r*cell_px
        draw.line([(0, y), (W*cell_px, y)], fill=_MS_GRID, width=1)
    for c in range(W+1):
        x = c*cell_px
        draw.line([(x, 0), (x, H*cell_px)], fill=_MS_GRID, width=1)

    # overlay de acciones válidas
    if show_valid_mask and env is not None:
        mask = np.asarray(jax.device_get(env.action_mask(state)[i]))  # (H*W,)
        mask = mask.reshape(H, W)
        # tint verdoso semitransparente
        overlay = Image.new("RGBA", img.size, (0,0,0,0))
        odraw = ImageDraw.Draw(overlay)
        for r in range(H):
            for c in range(W):
                if mask[r, c] and not revealed[r, c]:
                    x0, y0 = c*cell_px, r*cell_px
                    x1, y1 = x0+cell_px-1, y0+cell_px-1
                    odraw.rectangle([x0+1, y0+1, x1-1, y1-1], fill=_MS_MASK_VALID + (70,))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # highlight celda seleccionada
    if highlight_rc is not None:
        rr, cc = highlight_rc
        if 0 <= rr < H and 0 <= cc < W:
            x0, y0 = cc*cell_px, rr*cell_px
            x1, y1 = x0+cell_px-1, y0+cell_px-1
            draw.rectangle([x0+1, y0+1, x1-1, y1-1], outline=_MS_HILITE, width=3)

    return img

# ------------------------------ Panel "lo que ve el agente" -------------------
def window_patch_numbers(state: MinesState, i: int, rr: int, cc: int, R: int) -> np.ndarray:
    """(2R+1)x(2R+1) con números visibles; -1 si celda no revelada; pad cero fuera del tablero."""
    numbers  = np.asarray(jax.device_get(state.numbers[i]))
    revealed = np.asarray(jax.device_get(state.revealed[i]))
    H, W = revealed.shape
    vis = np.where(revealed, numbers, -1).astype(np.int16)
    # extrae ventana con padding
    r0, r1 = rr - R, rr + R + 1
    c0, c1 = cc - R, cc + R + 1
    out = -np.ones((2*R+1, 2*R+1), dtype=np.int16)
    for r in range(max(0, r0), min(H, r1)):
        for c in range(max(0, c0), min(W, c1)):
            out[r - r0, c - c0] = vis[r, c]
    return out

def render_small_grid(arr: np.ndarray, cell_px: int = 36) -> Image.Image:
    """Dibuja una matriz pequeña de enteros (-1=oculto) para el panel."""
    H, W = arr.shape
    img = Image.new("RGB", (W*cell_px, H*cell_px), (245,245,245))
    draw = ImageDraw.Draw(img)
    font = _safe_font(max(12, int(cell_px*0.6)))
    for r in range(H):
        for c in range(W):
            x0, y0 = c*cell_px, r*cell_px
            x1, y1 = x0+cell_px-1, y0+cell_px-1
            v = int(arr[r, c])
            if v < 0:
                # oculto
                draw.rectangle([x0, y0, x1, y1], fill=(200,200,200))
                txt = "■"
                tw, th = _measure_text(draw, txt, font)
                draw.text((x0 + (cell_px - tw)//2, y0 + (cell_px - th)//2 - 1),
                          txt, fill=(80,80,80), font=font)
            else:
                bg = _MS_BG_REVEALED if v > 0 else _MS_BG_ZERO
                draw.rectangle([x0, y0, x1, y1], fill=bg)
                if v > 0:
                    txt = str(v)
                    tw, th = _measure_text(draw, txt, font)
                    color = _MS_NUM_COLORS.get(v, (0,0,0))
                    draw.text((x0 + (cell_px - tw)//2, y0 + (cell_px - th)//2 - 1),
                              txt, fill=color, font=font)
            # grid
            draw.rectangle([x0, y0, x1, y1], outline=(180,180,180), width=1)
    return img

def get_summary_maps_from_observe(env: MinesweeperJAX, state: MinesState, i: int = 0):
    """Usa env.observe para recuperar los 2 mapas de resumen en las últimas capas:
       cnt_unrevealed, cnt_flags (ambos 'SAME' y sin contar el centro)."""
    obs = env.observe(state)  # (B, C, H, W)
    obs_np = np.asarray(jax.device_get(obs))
    cnt_unrev = obs_np[i, -2]  # penúltima capa
    cnt_flag  = obs_np[i, -1]  # última capa
    return cnt_unrev, cnt_flag

# ------------------------------ Streamlit App ---------------------------------
st.set_page_config(page_title="Minesweeper JAX – Interactivo", layout="wide")

# Sidebar: config
st.sidebar.title("Config")
H = st.sidebar.number_input("Altura H", min_value=4, max_value=64, value=16, step=1)
W = st.sidebar.number_input("Ancho W",  min_value=4, max_value=64, value=16, step=1)
mine_prob = st.sidebar.slider("Prob. mina", min_value=0.05, max_value=0.4, value=0.15625, step=0.01)
R = st.sidebar.slider("Context radius R", min_value=0, max_value=4, value=1, step=1)
cell_px = st.sidebar.slider("Tamaño celda (px)", min_value=20, max_value=48, value=28, step=2)
show_valid = st.sidebar.checkbox("Mostrar acciones válidas (overlay verde)", value=False)
mode = st.sidebar.selectbox("Modo de paso del agente", ["manual (click)", "first_valid", "random"])
btn_reset = st.sidebar.button("Resetear entorno")

# Estado
if "env" not in st.session_state or btn_reset or \
   st.session_state.get("H") != H or st.session_state.get("W") != W or \
   st.session_state.get("mine_prob") != mine_prob or \
   st.session_state.get("R") != R:
    key = jax.random.PRNGKey(0)
    env = MinesweeperJAX(H=int(H), W=int(W), mine_prob=float(mine_prob), context_radius=int(R))
    state = env.reset(key, batch_size=1)
    st.session_state.env = env
    st.session_state.state = state
    st.session_state.H, st.session_state.W = int(H), int(W)
    st.session_state.mine_prob = float(mine_prob)
    st.session_state.R = int(R)
    st.session_state.selected_rc = (H//2, W//2)  # selección por defecto
else:
    env = st.session_state.env
    state = st.session_state.state

st.title("Minesweeper JAX – Interactivo")
c1, c2 = st.columns([2.2, 1.0])

# --- Columna 1: Tablero + interacción
with c1:
    st.subheader("Tablero")
    rr, cc = st.session_state.get("selected_rc", (H//2, W//2))

    # Si el modo es 'first_valid' o 'random', un botón para ejecutar un paso
    if mode != "manual (click)":
        if st.button("Paso del agente"):
            mask = np.asarray(jax.device_get(env.action_mask(state)))
            if mode == "first_valid":
                a = mask.argmax(axis=1).astype(np.int32)
            else:
                # random entre válidas
                valid = np.where(mask[0])[0]
                pick = np.random.choice(valid) if len(valid) else 0
                a = np.array([pick], dtype=np.int32)
            state, r, d = env.step(state, jnp.asarray(a))
            st.session_state.state = state

    # Render del tablero actual
    board_img = render_board_img(state, i=0, cell_px=cell_px,
                                 highlight_rc=(rr, cc),
                                 show_valid_mask=show_valid,
                                 env=env)
    st.caption("Click en la imagen para seleccionar celda (en modo manual, revela).")
    click = streamlit_image_coordinates(board_img)
    # click es dict {'x':..., 'y':..., 'width':..., 'height':...}
    if click and "x" in click and "y" in click:
        c = int(click["x"] // cell_px)
        r = int(click["y"] // cell_px)
        # guarda selección
        st.session_state.selected_rc = (r, c)
        rr, cc = r, c
        # si estoy en modo manual, ejecutar paso con esa acción
        if mode == "manual (click)":
            a_idx = _rc_to_index(r, c, env.W)
            a = np.array([a_idx], dtype=np.int32)
            state, rwd, done = env.step(state, jnp.asarray(a))
            st.session_state.state = state
            # re-render
            board_img = render_board_img(state, i=0, cell_px=cell_px,
                                         highlight_rc=(rr, cc),
                                         show_valid_mask=show_valid,
                                         env=env)

    st.image(board_img, use_container_width=False)

# --- Columna 2: "lo que ve el agente"
with c2:
    st.subheader("Lo que ve el agente")
    st.write(f"Celda seleccionada: fila **{rr}**, col **{cc}**, radio **R={R}**")

    # Ventana local de números visibles (2R+1)x(2R+1)
    if R > 0:
        patch = window_patch_numbers(state, 0, rr, cc, R)
        st.caption("Ventana local de números visibles (–1 = oculto):")
        st.image(render_small_grid(patch, cell_px=34))

    # Mapas de resumen desde observe (cnt no-reveladas y banderas en ventana)
    if R > 0:
        cnt_unrev, cnt_flag = get_summary_maps_from_observe(env, state, i=0)
        # Mostrar el valor en la celda seleccionada y mini heatmaps
        H_, W_ = cnt_unrev.shape
        val_unrev = int(cnt_unrev[min(rr, H_-1), min(cc, W_-1)])
        val_flag  = int(cnt_flag[min(rr, H_-1),  min(cc, W_-1)])
        st.markdown(f"- **Conteo no-reveladas** en ventana (sin el centro): **{val_unrev}**")
        st.markdown(f"- **Conteo banderas** en ventana (sin el centro): **{val_flag}**")

        # para visualización rápida, normalizamos 0..max y pintamos escala gris
        def to_gray_img(a: np.ndarray) -> Image.Image:
            a = np.asarray(a)
            mmax = max(1.0, float(a.max()))
            g = (255.0 * (a / mmax)).astype(np.uint8)
            img = Image.fromarray(g, mode="L").convert("RGB").resize((W_*6, H_*6), Image.NEAREST)
            return img
        st.caption("Mini-mapas (grises): izquierda no-reveladas, derecha banderas.")
        st.image([to_gray_img(cnt_unrev), to_gray_img(cnt_flag)], use_container_width=True)

    # Acción válida?
    mask = np.asarray(jax.device_get(env.action_mask(state)[0])).reshape(env.H, env.W)
    st.markdown(f"¿Acción válida en ({rr},{cc})? **{'Sí' if mask[rr, cc] else 'No'}**")

st.divider()
st.caption("Tip: Esto usa `streamlit-image-coordinates` para click sobre imagen, y `jax.device_get` para traer buffers al host antes de dibujar.")
