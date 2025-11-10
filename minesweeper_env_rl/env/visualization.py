"""
Minesweeper visualization module with GIF generation.

Provides rendering functions similar to the uploaded code style.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import base64
from typing import Optional, Tuple, List
from minesweeper_pomdp import MinesweeperPOMDP

# Color scheme (Minesweeper classic style)
MS_BG_HIDDEN = (50, 50, 50)
MS_BG_REVEALED = (215, 215, 215)
MS_BG_ZERO = (230, 230, 230)
MS_FLAG = (240, 220, 70)
MS_GRID = (25, 25, 25)
MS_HILITE = (255, 160, 0)
MS_MINE = (200, 40, 40)

MS_NUM_COLORS = {
    1: (48, 130, 255),
    2: (64, 160, 64),
    3: (220, 60, 60),
    4: (64, 64, 160),
    5: (160, 64, 64),
    6: (64, 160, 160),
    7: (0, 0, 0),
    8: (128, 128, 128),
}


def _safe_font(px: int) -> ImageFont.FreeTypeFont:
    """Load a safe font, falling back to default if needed."""
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", px)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", px)
        except Exception:
            try:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", px)
            except Exception:
                return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, txt: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    """Measure text dimensions."""
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), txt, font=font)
        return (right - left), (bottom - top)
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(txt)
        return (right - left), (bottom - top)
    return font.getsize(txt)


def render_board_img(env: MinesweeperPOMDP,
                     cell_px: int = 28,
                     highlight_rc: Optional[Tuple[int, int]] = None,
                     show_mines: bool = False) -> Image.Image:
    """
    Render the Minesweeper board in classic style.
    
    Args:
        env: Minesweeper environment
        cell_px: Pixels per cell
        highlight_rc: Cell to highlight (row, col)
        show_mines: Show mine positions (for game over)
        
    Returns:
        PIL Image of the board
    """
    H, W = env.rows, env.cols
    img = Image.new("RGB", (W * cell_px, H * cell_px), MS_BG_HIDDEN)
    draw = ImageDraw.Draw(img)
    font = _safe_font(max(12, int(cell_px * 0.7)))
    
    # Draw cells
    for r in range(H):
        for c in range(W):
            x0, y0 = c * cell_px, r * cell_px
            x1, y1 = x0 + cell_px - 1, y0 + cell_px - 1
            
            # Flagged cell
            if env.flagged[r, c] and not env.revealed[r, c]:
                draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], fill=MS_FLAG)
                # Draw flag icon
                flag_points = [
                    (x0 + cell_px * 0.3, y0 + cell_px * 0.75),
                    (x0 + cell_px * 0.3, y0 + cell_px * 0.25),
                    (x0 + cell_px * 0.7, y0 + cell_px * 0.45)
                ]
                draw.polygon(flag_points, fill=MS_MINE)
                continue
            
            # Revealed cell
            if env.revealed[r, c]:
                # Check if it's a mine (game over)
                if env.mines[r, c]:
                    draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], fill=(180, 0, 0))
                    # Draw mine
                    center_x = x0 + cell_px // 2
                    center_y = y0 + cell_px // 2
                    radius = int(cell_px * 0.3)
                    draw.ellipse([center_x - radius, center_y - radius,
                                 center_x + radius, center_y + radius],
                                fill=(50, 50, 50))
                else:
                    bg = MS_BG_REVEALED if env.numbers[r, c] > 0 else MS_BG_ZERO
                    draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], fill=bg)
                    
                    n = int(env.numbers[r, c])
                    if n > 0:
                        txt = str(n)
                        tw, th = _measure_text(draw, txt, font)
                        cx = x0 + (cell_px - tw) // 2
                        cy = y0 + (cell_px - th) // 2 - 1
                        draw.text((cx, cy), txt, fill=MS_NUM_COLORS.get(n, (0, 0, 0)), font=font)
            
            # Hidden cell - show mine if requested
            elif show_mines and env.mines[r, c]:
                draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], fill=(100, 100, 100))
                txt = "M"
                tw, th = _measure_text(draw, txt, font)
                cx = x0 + (cell_px - tw) // 2
                cy = y0 + (cell_px - th) // 2 - 1
                draw.text((cx, cy), txt, fill=(200, 40, 40), font=font)
    
    # Draw grid lines
    for r in range(H + 1):
        y = r * cell_px
        draw.line([(0, y), (W * cell_px, y)], fill=MS_GRID, width=1)
    for c in range(W + 1):
        x = c * cell_px
        draw.line([(x, 0), (x, H * cell_px)], fill=MS_GRID, width=1)
    
    # Highlight cell if specified
    if highlight_rc is not None:
        rr, cc = highlight_rc
        if 0 <= rr < H and 0 <= cc < W:
            x0, y0 = cc * cell_px, rr * cell_px
            x1, y1 = x0 + cell_px - 1, y0 + cell_px - 1
            draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], outline=MS_HILITE, width=3)
    
    return img


def render_window_img(window: np.ndarray, cell_px: int = 32) -> Image.Image:
    """
    Render a local window observation.
    
    Args:
        window: Window array (-1=hidden, 0-8=number, 9=flagged)
        cell_px: Pixels per cell
        
    Returns:
        PIL Image of the window
    """
    H, W = window.shape
    img = Image.new("RGB", (W * cell_px, H * cell_px), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = _safe_font(max(12, int(cell_px * 0.6)))
    
    for r in range(H):
        for c in range(W):
            x0, y0 = c * cell_px, r * cell_px
            x1, y1 = x0 + cell_px - 1, y0 + cell_px - 1
            v = int(window[r, c])
            
            if v < 0:  # Hidden
                draw.rectangle([x0, y0, x1, y1], fill=(200, 200, 200))
                txt = "■"
                tw, th = _measure_text(draw, txt, font)
                draw.text((x0 + (cell_px - tw) // 2, y0 + (cell_px - th) // 2 - 1),
                         txt, fill=(80, 80, 80), font=font)
            
            elif v == 9:  # Flagged
                draw.rectangle([x0, y0, x1, y1], fill=MS_FLAG)
                txt = "F"
                tw, th = _measure_text(draw, txt, font)
                draw.text((x0 + (cell_px - tw) // 2, y0 + (cell_px - th) // 2 - 1),
                         txt, fill=(200, 40, 40), font=font)
            
            else:  # Revealed (0-8)
                bg = MS_BG_REVEALED if v > 0 else MS_BG_ZERO
                draw.rectangle([x0, y0, x1, y1], fill=bg)
                if v > 0:
                    txt = str(v)
                    tw, th = _measure_text(draw, txt, font)
                    color = MS_NUM_COLORS.get(v, (0, 0, 0))
                    draw.text((x0 + (cell_px - tw) // 2, y0 + (cell_px - th) // 2 - 1),
                             txt, fill=color, font=font)
            
            # Border
            draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)
    
    return img


def combine_board_and_window(board_img: Image.Image, 
                            window_img: Image.Image, 
                            sep: int = 8) -> Image.Image:
    """
    Combine board and window images horizontally.
    
    Args:
        board_img: Main board image
        window_img: Local window image
        sep: Separation pixels
        
    Returns:
        Combined image
    """
    h = max(board_img.height, window_img.height)
    w = board_img.width + sep + window_img.width
    canvas = Image.new("RGB", (w, h), (20, 20, 20))
    
    # Paste images centered vertically
    canvas.paste(board_img, (0, (h - board_img.height) // 2))
    canvas.paste(window_img, (board_img.width + sep, (h - window_img.height) // 2))
    
    return canvas


def add_text_overlay(img: Image.Image, text: str, position: str = "top") -> Image.Image:
    """
    Add text overlay to image.
    
    Args:
        img: Base image
        text: Text to add
        position: "top" or "bottom"
        
    Returns:
        Image with text overlay
    """
    # Create new image with extra space for text
    text_height = 30
    if position == "top":
        new_img = Image.new("RGB", (img.width, img.height + text_height), (20, 20, 20))
        new_img.paste(img, (0, text_height))
        text_y = 5
    else:
        new_img = Image.new("RGB", (img.width, img.height + text_height), (20, 20, 20))
        new_img.paste(img, (0, 0))
        text_y = img.height + 5
    
    draw = ImageDraw.Draw(new_img)
    font = _safe_font(16)
    
    # Center text
    tw, th = _measure_text(draw, text, font)
    text_x = (new_img.width - tw) // 2
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    return new_img


def create_episode_gif(env: MinesweeperPOMDP,
                      output_path: str = "episode.gif",
                      cell_px: int = 28,
                      duration: float = 0.5,
                      show_windows: bool = True,
                      window_cell_px: int = 32) -> str:
    """
    Create animated GIF from episode history.
    
    Args:
        env: Minesweeper environment with history
        output_path: Path to save GIF
        cell_px: Pixels per cell for main board
        duration: Duration per frame in seconds
        show_windows: Show local windows for each action
        window_cell_px: Pixels per cell for windows
        
    Returns:
        Path to saved GIF
    """
    frames = []
    
    for i, hist in enumerate(env.history):
        # Create temporary environment state for this frame
        temp_env = env.clone()
        temp_env.revealed = hist['revealed']
        temp_env.flagged = hist['flagged']
        
        # Determine highlight cell
        highlight = None
        if hist['action'] is not None:
            highlight = hist['action']['cell']
        
        # Render board
        board_img = render_board_img(temp_env, cell_px=cell_px, highlight_rc=highlight,
                                    show_mines=(i == len(env.history) - 1 and env.done))
        
        # Add window if action exists and requested
        if show_windows and hist['action'] is not None and 'local_window' in hist['action']:
            window = hist['action'].get('local_window')
            if window is not None:
                window_img = render_window_img(window, cell_px=window_cell_px)
                frame = combine_board_and_window(board_img, window_img)
            else:
                frame = board_img
        else:
            frame = board_img
        
        # Add text overlay
        action_type = hist.get('action_type', 'Initial')
        reward = hist.get('reward', 0.0)
        text = f"Step {hist['step']}: {action_type} | Reward: {reward:.2f}"
        
        if i == len(env.history) - 1 and env.done:
            if env.won:
                text += " | WON! 🎉"
            else:
                text += " | LOST 💥"
        
        frame = add_text_overlay(frame, text, position="top")
        
        frames.append(frame)
    
    # Save GIF
    if frames:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        duration_ms = int(duration * 1000)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,  # Infinite loop
            disposal=2
        )
        print(f"GIF saved to {output_path}")
    
    return output_path


def create_step_images(env: MinesweeperPOMDP,
                      output_dir: str = "steps",
                      cell_px: int = 28) -> List[str]:
    """
    Create individual images for each step.
    
    Args:
        env: Minesweeper environment with history
        output_dir: Directory to save images
        cell_px: Pixels per cell
        
    Returns:
        List of saved image paths
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    
    for i, hist in enumerate(env.history):
        temp_env = env.clone()
        temp_env.revealed = hist['revealed']
        temp_env.flagged = hist['flagged']
        
        highlight = None
        if hist['action'] is not None:
            highlight = hist['action']['cell']
        
        show_mines = (i == len(env.history) - 1 and env.done)
        img = render_board_img(temp_env, cell_px=cell_px, 
                              highlight_rc=highlight, show_mines=show_mines)
        
        # Add text
        action_type = hist.get('action_type', 'Initial')
        text = f"Step {hist['step']}: {action_type}"
        img = add_text_overlay(img, text, position="top")
        
        # Save
        path = os.path.join(output_dir, f"step_{i:03d}.png")
        img.save(path)
        paths.append(path)
    
    print(f"Saved {len(paths)} images to {output_dir}/")
    return paths


def display_gif_inline(gif_path: str):
    """
    Display GIF inline (for Jupyter notebooks).
    
    Args:
        gif_path: Path to GIF file
    """
    try:
        from IPython.display import HTML, display
        with open(gif_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        html = f'<img src="data:image/gif;base64,{data}" alt="episode gif" style="max-width: 100%;"/>'
        display(HTML(html))
    except ImportError:
        print("IPython not available. Cannot display inline.")
        print(f"GIF saved to: {gif_path}")


# Example usage
if __name__ == "__main__":
    import random
    from minesweeper_pomdp import MinesweeperPOMDP
    
    print("=== Minesweeper Visualization Demo ===\n")
    
    # Create environment
    env = MinesweeperPOMDP(rows=8, cols=8, num_mines=10, 
                          window_radius=2, window_shape="rhombus")
    
    # Reset and play
    obs = env.reset(first_click=(4, 4))
    
    # First move
    action = {'cell': (4, 4), 'type': 'reveal'}
    obs, reward, done, info = env.step(action)
    
    # Play a few random moves
    for _ in range(10):
        if done:
            break
        
        valid_actions = obs['valid_actions']
        reveal_actions = [a for a in valid_actions if a['type'] == 'reveal']
        
        if not reveal_actions:
            break
        
        action = random.choice(reveal_actions)
        
        # Store window in action for visualization
        if 'local_window' in obs:
            action['local_window'] = obs['local_window']
        
        obs, reward, done, info = env.step(action)
    
    print(f"Game finished in {env.step_count} steps")
    print(f"Won: {env.won}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Single frame
    print("Rendering final board...")
    final_img = render_board_img(env, cell_px=32, show_mines=True)
    final_img.save("/mnt/user-data/outputs/final_board.png")
    print("✓ Saved final_board.png")
    
    # GIF of episode
    print("\nCreating episode GIF...")
    gif_path = create_episode_gif(env, 
                                  output_path="/mnt/user-data/outputs/episode.gif",
                                  cell_px=24,
                                  duration=0.5,
                                  show_windows=False)
    print(f"✓ Created {gif_path}")
    
    # Step images
    print("\nCreating step images...")
    create_step_images(env, output_dir="/mnt/user-data/outputs/steps", cell_px=28)
    
    print("\n✓ Visualization complete!")

__all__ = [
    "render_board_img",
    "render_window_img",
    "combine_board_and_window",
    "add_text_overlay",
    "create_episode_gif",
    "create_step_images",
    "display_gif_inline",
]
