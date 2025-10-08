# record.py
# Genera un video MP4 de un episodio de Buscaminas usando tu entorno JAX.
# Depende de: pillow, imageio, imageio-ffmpeg, matplotlib, jax

from __future__ import annotations
import os as _os
import numpy as _np
import imageio.v2 as _iio
from PIL import Image, ImageDraw, ImageFont

import jax
import jax.numpy as jnp

# Importa tu entorno (usa la clase que ya pegaste en env/__init__.py)
# Asegúrate de que env/__init__.py esté en el PYTHONPATH (ejecuta desde la raíz del repo).
from env import MinesweeperJAX, MinesState

# ------------------------------ Paleta / estilos ------------------------------
_MS_BG_HIDDEN   = (50, 50, 50)
_MS_BG_REVEALED = (215, 215, 215)
_MS_BG_ZERO     = (230, 230, 230)
_MS_FLAG        = (240, 220, 70)
_MS_GRID        = (25, 25, 25)
_MS_BOMB        = (200, 20, 20)
_MS_HILITE      = (255, 160, 0)

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
    # Intenta una fuente mono; si no, usa la default
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", px)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", px)
        except Exception:
            return ImageFont.load_default()

def _measure_text(draw: ImageDraw.ImageDraw, txt: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Mide texto sin usar draw.textsize (eliminado en Pillow 10).
    Prioriza draw.textbbox (>=10.1), luego font.getbbox, y como último recurso
    draw.textlength (ancho) + altura aproximada por font.size.
    """
    # Pillow >=10.1: textbbox
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), txt, font=font)
        return (right - left), (bottom - top)
    # Compatibilidad: font.getbbox (Pillow >=8)
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(txt)
        return (right - left), (bottom - top)
    # Fallback: textlength para ancho + altura aproximada
    w = int(draw.textlength(txt, font=font)) if hasattr(draw, "textlength") else font.getsize(txt)[0]
    h = getattr(font, "size", 14)
    return w, h

def _index_to_rc(idx: int, W: int):
    return int(idx // W), int(idx % W)

def render_frame_ui(state: MinesState, i: int = 0, cell_px: int = 28,
                    highlight_action: int | None = None,
                    explode_mask: _np.ndarray | None = None) -> _np.ndarray:
    """
    Dibuja un frame (H*cell_px, W*cell_px, 3) con look de Buscaminas.
    - Las minas NO se muestran a menos que 'explode_mask' lo indique (explosión).
    - 'highlight_action' pinta un borde naranja en la celda elegida (pre-jugada).
    """
    # Traer arrays del device al host
    mines    = _np.asarray(jax.device_get(state.mines[i]))
    revealed = _np.asarray(jax.device_get(state.revealed[i]))
    flagged  = _np.asarray(jax.device_get(state.flagged[i]))
    numbers  = _np.asarray(jax.device_get(state.numbers[i]))

    H, W = revealed.shape
    img = Image.new("RGB", (W*cell_px, H*cell_px), _MS_BG_HIDDEN)
    draw = ImageDraw.Draw(img)
    font = _safe_font(max(12, int(cell_px*0.7)))

    # Cuadrícula
    for r in range(H+1):
        y = r*cell_px
        draw.line([(0, y), (W*cell_px, y)], fill=_MS_GRID, width=1)
    for c in range(W+1):
        x = c*cell_px
        draw.line([(x, 0), (x, H*cell_px)], fill=_MS_GRID, width=1)

    # Celdas
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
            else:
                # Oculta (ya tiene fondo hidden)
                pass

    # Mostrar minas SOLO si explotó esa celda (explode_mask)
    if explode_mask is not None:
        em = _np.asarray(explode_mask, dtype=bool)
        for r in range(H):
            for c in range(W):
                if em[r, c]:
                    x0, y0 = c*cell_px, r*cell_px
                    x1, y1 = x0+cell_px-1, y0+cell_px-1
                    draw.rectangle([x0+1, y0+1, x1-1, y1-1], fill=_MS_BOMB)
                    draw.line([(x0+4, y0+4), (x1-4, y1-4)], fill=(0,0,0), width=2)
                    draw.line([(x1-4, y0+4), (x0+4, y1-4)], fill=(0,0,0), width=2)

    # Resalta la acción elegida (borde naranja)
    if highlight_action is not None:
        rr, cc = _index_to_rc(int(highlight_action), W)
        x0, y0 = cc*cell_px, rr*cell_px
        x1, y1 = x0+cell_px-1, y0+cell_px-1
        draw.rectangle([x0+1, y0+1, x1-1, y1-1], outline=_MS_HILITE, width=3)

    return _np.asarray(img, dtype=_np.uint8)

def _choose_action_first_valid(env: MinesweeperJAX, state: MinesState) -> _np.ndarray:
    mask = _np.asarray(jax.device_get(env.action_mask(state)))
    return mask.argmax(axis=1).astype(_np.int32)

def _choose_action_random_valid(env: MinesweeperJAX, state: MinesState) -> _np.ndarray:
    mask = _np.asarray(jax.device_get(env.action_mask(state)))
    a = _np.zeros((mask.shape[0],), dtype=_np.int32)
    for b in range(mask.shape[0]):
        valid = _np.where(mask[b])[0]
        a[b] = _np.random.choice(valid) if len(valid) else 0
    return a

def record_episode_video(env: MinesweeperJAX,
                         init_state: MinesState,
                         out_path: str = "videos/mines_episode.mp4",
                         fps: int = 8,
                         cell_px: int = 28,
                         policy: str = "first_valid",
                         max_steps: int = 200,
                         batch_index: int = 0):
    """
    Graba un episodio como MP4 mostrando:
      • Frame PRE-jugada (resalta celda elegida)
      • Frame POST-jugada (muestra números; revela mina SOLO si explotó)
    """
    _os.makedirs(_os.path.dirname(out_path) or ".", exist_ok=True)
    writer = _iio.get_writer(out_path, fps=fps)  # usa imageio-ffmpeg
    try:
        state = init_state
        B = int(_np.asarray(jax.device_get(state.revealed.shape[0])))
        assert 0 <= batch_index < B, "batch_index fuera de rango"

        for t in range(max_steps):
            print(f"Step {t+1}/{max_steps}", end="\r")
            # Política
            if policy == "random":
                a = _choose_action_random_valid(env, state)
            else:
                a = _choose_action_first_valid(env, state)
            a_idx = int(a[batch_index])

            # PRE: highlight
            pre_frame = render_frame_ui(state, i=batch_index, cell_px=cell_px,
                                        highlight_action=a_idx, explode_mask=None)
            writer.append_data(pre_frame)

            # Saber si explota
            H, W = state.revealed.shape[1], state.revealed.shape[2]
            rr, cc = _index_to_rc(a_idx, W)
            mines    = _np.asarray(jax.device_get(state.mines[batch_index]))
            revealed = _np.asarray(jax.device_get(state.revealed[batch_index]))
            flagged  = _np.asarray(jax.device_get(state.flagged[batch_index]))
            will_boom = (not revealed[rr, cc]) and (not flagged[rr, cc]) and bool(mines[rr, cc])

            # STEP
            state, r, d = env.step(state, jnp.asarray(a))

            # POST
            explode_mask = None
            if will_boom:
                em = _np.zeros_like(mines, dtype=bool)
                em[rr, cc] = True
                explode_mask = em

            post_frame = render_frame_ui(state, i=batch_index, cell_px=cell_px,
                                         highlight_action=None, explode_mask=explode_mask)
            writer.append_data(post_frame)

            if bool(_np.asarray(jax.device_get(d[batch_index]))):
                break
    finally:
        writer.close()
    print(f"[OK] Video guardado en: {out_path}")

# --------------------------------- Main ---------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    env = MinesweeperJAX(H=16, W=16, mine_prob=0.15625, context_radius=1)
    # Para video, reset sin jit (suficiente y más flexible)
    state = env.reset(key, batch_size=1)

    record_episode_video(env, state,
                         out_path="videos/mines_episode.mp4",
                         fps=8,
                         cell_px=30,
                         policy="random",  # o "random"
                         max_steps=300,
                         batch_index=0)
