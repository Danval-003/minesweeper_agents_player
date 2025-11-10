"""
Optimized Minesweeper POMDP (Partially Observable MDP) with windowed observations.

Features:
- Efficient state management with internal memory
- Partial observations (local windows around cells)
- Rhombus/circular window shapes
- Flag/unflag actions
- Fast batch updates
- No redundant cell-by-cell processing
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Set
from dataclasses import dataclass
from collections import deque
import copy

@dataclass
class CellState:
    """Efficient cell state representation."""
    revealed: bool = False
    flagged: bool = False
    is_mine: bool = False
    adjacent_mines: int = 0


class MinesweeperPOMDP:
    """
    Optimized Minesweeper environment with partial observability.
    
    Key optimizations:
    - Internal state cache for fast lookups
    - Batch reveal operations
    - Efficient window extraction
    - No redundant iterations
    """
    
    def __init__(self, 
                 rows: int = 8, 
                 cols: int = 8, 
                 num_mines: int = 10,
                 window_radius: int = 2,
                 window_shape: str = "rhombus"):
        """
        Initialize Minesweeper POMDP.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            num_mines: Number of mines
            window_radius: Radius of observation window
            window_shape: "rhombus", "circle", or "square"
        """
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.window_radius = window_radius
        self.window_shape = window_shape
        
        # Precompute window offsets for efficiency
        self._window_offsets = self._precompute_window_offsets()
        
        # State storage
        self.grid: np.ndarray = None  # Internal grid state
        self.revealed: np.ndarray = None  # Boolean mask
        self.flagged: np.ndarray = None  # Boolean mask
        self.numbers: np.ndarray = None  # Adjacent mine counts
        self.mines: np.ndarray = None  # Mine positions
        
        # Episode tracking
        self.done = False
        self.won = False
        self.step_count = 0
        self.total_safe_cells = rows * cols - num_mines
        self.revealed_count = 0
        
        # History for visualization
        self.history: List[Dict] = []
        
    def _precompute_window_offsets(self) -> List[Tuple[int, int]]:
        """
        Precompute offsets for window shape (rhombus, circle, or square).
        This is computed once for efficiency.
        """
        R = self.window_radius
        offsets = []
        
        if self.window_shape == "rhombus":
            # Manhattan distance <= R
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    if abs(dr) + abs(dc) <= R:
                        offsets.append((dr, dc))
                        
        elif self.window_shape == "circle":
            # Euclidean distance <= R
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    if dr*dr + dc*dc <= R*R:
                        offsets.append((dr, dc))
                        
        else:  # square
            # Max distance <= R
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    offsets.append((dr, dc))
        
        return offsets
    
    def reset(self, first_click: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Reset environment.
        
        Args:
            first_click: Optional guaranteed safe first click
            
        Returns:
            Initial observation dictionary
        """
        self.step_count = 0
        self.done = False
        self.won = False
        self.revealed_count = 0
        self.history = []
        
        # Initialize grids (vectorized)
        self.revealed = np.zeros((self.rows, self.cols), dtype=np.bool_)
        self.flagged = np.zeros((self.rows, self.cols), dtype=np.bool_)
        self.mines = np.zeros((self.rows, self.cols), dtype=np.bool_)
        
        # Place mines efficiently
        self._place_mines_fast(first_click)
        
        # Calculate adjacent mines (vectorized)
        self.numbers = self._calculate_numbers_vectorized()
        
        # Initial observation
        obs = self._get_observation()
        
        # Record history
        self.history.append({
            'step': 0,
            'action': None,
            'action_type': None,
            'revealed': self.revealed.copy(),
            'flagged': self.flagged.copy(),
            'reward': 0.0
        })
        
        return obs
    
    def _place_mines_fast(self, safe_cell: Optional[Tuple[int, int]] = None):
        """Efficiently place mines using vectorized operations."""
        # Create flat index array
        total_cells = self.rows * self.cols
        indices = np.arange(total_cells)
        
        # Remove safe zone if specified
        if safe_cell is not None:
            r, c = safe_cell
            # Get safe zone (cell + neighbors)
            safe_indices = set()
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        safe_indices.add(nr * self.cols + nc)
            
            # Remove safe indices
            indices = np.array([i for i in indices if i not in safe_indices])
        
        # Randomly select mine positions
        mine_indices = np.random.choice(indices, size=self.num_mines, replace=False)
        
        # Convert to 2D coordinates and set mines
        mine_rows = mine_indices // self.cols
        mine_cols = mine_indices % self.cols
        self.mines[mine_rows, mine_cols] = True
    
    def _calculate_numbers_vectorized(self) -> np.ndarray:
        """
        Calculate adjacent mine counts using convolution for speed.
        This is much faster than nested loops.
        """
        from scipy.signal import convolve2d
        
        # Kernel for counting neighbors
        kernel = np.ones((3, 3), dtype=np.int32)
        kernel[1, 1] = 0
        
        # Convolve to count adjacent mines
        numbers = convolve2d(self.mines.astype(np.int32), kernel, mode='same')
        
        # Set mine cells to -1 (optional, for clarity)
        # numbers[self.mines] = -1
        
        return numbers.astype(np.int32)
    
    def _get_observation(self, cell: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Get observation dictionary.
        
        Args:
            cell: If provided, returns windowed observation around this cell
            
        Returns:
            Dictionary containing global and/or local observations
        """
        obs = {
            'global_state': self._get_global_observation(),
            'valid_actions': self._get_valid_actions_fast()
        }
        
        if cell is not None:
            obs['local_window'] = self._get_window_observation(cell)
        
        # Add metadata
        obs['step'] = self.step_count
        obs['revealed_count'] = self.revealed_count
        obs['flag_count'] = np.sum(self.flagged)
        obs['done'] = self.done
        
        return obs
    
    def _get_global_observation(self) -> np.ndarray:
        """
        Get global state observation (3 channels).
        
        Returns:
            Array of shape (rows, cols, 3):
            - Channel 0: Revealed mask
            - Channel 1: Numbers (or -1 if not revealed)
            - Channel 2: Flagged mask
        """
        obs = np.zeros((self.rows, self.cols, 3), dtype=np.float32)
        obs[:, :, 0] = self.revealed.astype(np.float32)
        obs[:, :, 1] = np.where(self.revealed, self.numbers, -1).astype(np.float32)
        obs[:, :, 2] = self.flagged.astype(np.float32)
        return obs
    
    def _get_window_observation(self, cell: Tuple[int, int]) -> np.ndarray:
        """
        Get windowed observation around a cell using precomputed offsets.
        
        Args:
            cell: (row, col) center of window
            
        Returns:
            Array representing local window (values: -1=hidden, 0-8=revealed, 9=flagged)
        """
        r, c = cell
        window_size = 2 * self.window_radius + 1
        window = -np.ones((window_size, window_size), dtype=np.int32)
        
        # Use precomputed offsets for efficiency
        for dr, dc in self._window_offsets:
            nr, nc = r + dr, c + dc
            
            # Check bounds
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                # Map to window coordinates
                wr = dr + self.window_radius
                wc = dc + self.window_radius
                
                if self.flagged[nr, nc]:
                    window[wr, wc] = 9  # Special value for flagged
                elif self.revealed[nr, nc]:
                    window[wr, wc] = self.numbers[nr, nc]
                # else: remains -1 (hidden)
        
        return window
    
    def _get_valid_actions_fast(self) -> List[Dict]:
        """
        Get valid actions efficiently.
        
        Returns:
            List of action dictionaries with 'cell' and 'type'
        """
        actions = []
        
        # Vectorized: find unrevealed cells
        unrevealed_mask = ~self.revealed
        unrevealed_coords = np.argwhere(unrevealed_mask)
        
        for r, c in unrevealed_coords:
            # Can click if not flagged
            if not self.flagged[r, c]:
                actions.append({'cell': (r, c), 'type': 'reveal'})
            
            # Can always toggle flag on unrevealed cells
            flag_action = 'unflag' if self.flagged[r, c] else 'flag'
            actions.append({'cell': (r, c), 'type': flag_action})
        
        return actions
    
    def _reveal_cell_fast(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """
        Reveal cell and flood-fill zeros efficiently using BFS.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Set of newly revealed cells
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return set()
        
        if self.revealed[row, col] or self.flagged[row, col]:
            return set()
        
        newly_revealed = set()
        queue = deque([(row, col)])
        visited = {(row, col)}
        
        while queue:
            r, c = queue.popleft()
            
            # Skip if already revealed or flagged
            if self.revealed[r, c] or self.flagged[r, c]:
                continue
            
            # Reveal cell
            self.revealed[r, c] = True
            newly_revealed.add((r, c))
            self.revealed_count += 1
            
            # If it's a zero, add neighbors to queue
            if self.numbers[r, c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.rows and 0 <= nc < self.cols and
                            (nr, nc) not in visited and
                            not self.revealed[nr, nc] and
                            not self.flagged[nr, nc]):
                            queue.append((nr, nc))
                            visited.add((nr, nc))
        
        return newly_revealed
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Dictionary with 'cell' (row, col) and 'type' ('reveal', 'flag', 'unflag')
            
        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self._get_observation(), 0.0, True, {'error': 'Episode already done'}
        
        cell = action['cell']
        action_type = action['type']
        row, col = cell
        
        # Validate action
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return self._get_observation(), -0.1, False, {'error': 'Invalid cell'}
        
        reward = 0.0
        info = {'action': action}
        
        if action_type == 'reveal':
            # Cannot reveal if flagged
            if self.flagged[row, col]:
                return self._get_observation(), -0.05, False, {'error': 'Cell is flagged'}
            
            # Already revealed
            if self.revealed[row, col]:
                return self._get_observation(), -0.05, False, {'error': 'Already revealed'}
            
            # Check if mine
            if self.mines[row, col]:
                self.revealed[row, col] = True
                self.done = True
                self.won = False
                reward = -1.0
                info['hit_mine'] = True
            else:
                # Reveal cell(s) efficiently
                newly_revealed = self._reveal_cell_fast(row, col)
                cells_revealed = len(newly_revealed)
                reward = cells_revealed * 0.1
                info['cells_revealed'] = cells_revealed
                
                # Check win condition
                if self.revealed_count >= self.total_safe_cells:
                    self.done = True
                    self.won = True
                    reward += 1.0
                    info['won'] = True
        
        elif action_type == 'flag':
            if not self.revealed[row, col] and not self.flagged[row, col]:
                self.flagged[row, col] = True
                reward = 0.01  # Small reward for flagging
                info['flagged'] = True
            else:
                reward = -0.05
                info['error'] = 'Cannot flag'
        
        elif action_type == 'unflag':
            if self.flagged[row, col]:
                self.flagged[row, col] = False
                reward = 0.0
                info['unflagged'] = True
            else:
                reward = -0.05
                info['error'] = 'Cell not flagged'
        
        else:
            return self._get_observation(), -0.1, False, {'error': 'Invalid action type'}
        
        self.step_count += 1
        
        # Record history
        self.history.append({
            'step': self.step_count,
            'action': action,
            'action_type': action_type,
            'revealed': self.revealed.copy(),
            'flagged': self.flagged.copy(),
            'reward': reward
        })
        
        # Get observation with window around acted cell
        obs = self._get_observation(cell=cell)
        
        return obs, reward, self.done, info
    
    def get_state_summary(self) -> Dict:
        """Get summary statistics of current state."""
        return {
            'step': self.step_count,
            'revealed': int(self.revealed_count),
            'total_safe': self.total_safe_cells,
            'flagged': int(np.sum(self.flagged)),
            'mines': self.num_mines,
            'done': self.done,
            'won': self.won,
            'completion': self.revealed_count / self.total_safe_cells * 100
        }
    
    def render_text(self, show_mines: bool = False) -> str:
        """
        Render board as text.
        
        Args:
            show_mines: Show mine positions (for debugging)
            
        Returns:
            String representation of board
        """
        lines = []
        lines.append(f"\nStep {self.step_count} | Revealed: {self.revealed_count}/{self.total_safe_cells}")
        lines.append("   " + " ".join(f"{c:2}" for c in range(self.cols)))
        
        for r in range(self.rows):
            row_str = f"{r:2} "
            for c in range(self.cols):
                if self.flagged[r, c]:
                    row_str += " F"
                elif self.revealed[r, c]:
                    if self.mines[r, c]:
                        row_str += " *"
                    else:
                        n = self.numbers[r, c]
                        row_str += f" {n}" if n > 0 else " ."
                else:
                    if show_mines and self.mines[r, c]:
                        row_str += " M"
                    else:
                        row_str += " #"
            lines.append(row_str)
        
        return "\n".join(lines)
    
    def clone(self) -> 'MinesweeperPOMDP':
        """Create a deep copy of the environment."""
        env_copy = MinesweeperPOMDP(
            self.rows, self.cols, self.num_mines,
            self.window_radius, self.window_shape
        )
        
        # Copy all state
        env_copy.revealed = self.revealed.copy()
        env_copy.flagged = self.flagged.copy()
        env_copy.mines = self.mines.copy()
        env_copy.numbers = self.numbers.copy()
        env_copy.done = self.done
        env_copy.won = self.won
        env_copy.step_count = self.step_count
        env_copy.revealed_count = self.revealed_count
        env_copy.history = copy.deepcopy(self.history)
        
        return env_copy


# Example usage
if __name__ == "__main__":
    import random
    
    print("=== Optimized Minesweeper POMDP Demo ===\n")
    
    # Create environment with rhombus window
    env = MinesweeperPOMDP(
        rows=8, 
        cols=8, 
        num_mines=10,
        window_radius=2,
        window_shape="rhombus"
    )
    
    print(f"Grid: {env.rows}x{env.cols}")
    print(f"Mines: {env.num_mines}")
    print(f"Window: {env.window_shape} with radius {env.window_radius}")
    print(f"Window cells: {len(env._window_offsets)}")
    
    # Reset
    obs = env.reset(first_click=(4, 4))
    
    print("\n" + env.render_text())
    
    # Make first move
    action = {'cell': (4, 4), 'type': 'reveal'}
    obs, reward, done, info = env.step(action)
    
    print(f"\nAction: Reveal (4, 4)")
    print(f"Reward: {reward:.2f}")
    print(f"Cells revealed: {info.get('cells_revealed', 0)}")
    print("\n" + env.render_text())
    
    # Show local window
    if 'local_window' in obs:
        print("\nLocal window around (4, 4):")
        window = obs['local_window']
        for r in range(window.shape[0]):
            row_str = ""
            for c in range(window.shape[1]):
                v = window[r, c]
                if v == -1:
                    row_str += " #"
                elif v == 9:
                    row_str += " F"
                else:
                    row_str += f" {v}" if v > 0 else " ."
            print(row_str)
    
    # Play a few more moves
    for i in range(5):
        valid_actions = obs['valid_actions']
        reveal_actions = [a for a in valid_actions if a['type'] == 'reveal']
        
        if not reveal_actions or done:
            break
        
        action = random.choice(reveal_actions)
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {env.step_count}: {action['type']} {action['cell']}")
        print(f"Reward: {reward:.2f}")
        
        if done:
            print(env.render_text(show_mines=True))
            if env.won:
                print("\n🎉 Won!")
            else:
                print("\n💥 Hit a mine!")
            break
    
    # Show statistics
    print("\n" + "="*50)
    summary = env.get_state_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

__all__ = ["MinesweeperPOMDP", "CellState"]