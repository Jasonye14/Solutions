import numpy as np
from PIL import Image
import os

def read_starting_position(file_path):
    """
    Read the starting position image and convert it to a 2D numpy array.
    White pixels (255) represent live cells, black pixels (0) represent dead cells.
    
    Args:
        file_path (str): Path to the starting_position.png file
        
    Returns:
        numpy.ndarray: 2D boolean array where True represents live cells and False represents dead cells
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Starting position file not found: {file_path}")
    
    # Read image using PIL
    with Image.open(file_path) as img:
        # Convert to grayscale
        img_gray = img.convert('L')
        # Convert to numpy array
        grid = np.array(img_gray)
        # Convert to boolean array (True for live cells, False for dead cells)
        # White (255) becomes True, Black (0) becomes False
        grid = grid > 127
    
    return grid

def count_neighbors(grid):
    """
    Count the number of live neighbors for each cell in the grid.
    
    Args:
        grid (numpy.ndarray): 2D boolean array representing the current state
        
    Returns:
        numpy.ndarray: 2D integer array with the count of live neighbors for each cell
    """
    rows, cols = grid.shape
    # Create a zero-padded version of the grid to handle edges
    padded = np.pad(grid, pad_width=1, mode='constant', constant_values=False)
    
    # Initialize neighbor count array
    neighbors = np.zeros((rows, cols), dtype=int)
    
    # Count neighbors using array slicing
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:  # Skip the cell itself
                continue
            neighbors += padded[i:i+rows, j:j+cols]
    
    return neighbors

def next_generation(grid):
    """
    Compute the next generation of the Game of Life based on the current state.
    
    Rules:
    1. Any live cell with fewer than two live neighbors dies (underpopulation)
    2. Any live cell with two or three live neighbors lives on
    3. Any live cell with more than three live neighbors dies (overpopulation)
    4. Any dead cell with exactly three live neighbors becomes alive (reproduction)
    
    Args:
        grid (numpy.ndarray): 2D boolean array representing the current state
        
    Returns:
        numpy.ndarray: 2D boolean array representing the next generation
    """
    neighbor_counts = count_neighbors(grid)
    
    # Apply the rules
    new_grid = np.zeros_like(grid)
    
    # Rule 1 & 2 & 3: Live cells survive if they have 2 or 3 neighbors
    survival_mask = grid & ((neighbor_counts == 2) | (neighbor_counts == 3))
    
    # Rule 4: Dead cells come alive if they have exactly 3 neighbors
    birth_mask = ~grid & (neighbor_counts == 3)
    
    new_grid = survival_mask | birth_mask
    return new_grid

def save_grid_as_image(grid, output_path):
    """
    Save the grid state as a PNG image.
    
    Args:
        grid (numpy.ndarray): 2D boolean array representing the game state
        output_path (str): Path where to save the PNG file
    """
    # Convert boolean array to uint8 array (False->0, True->255)
    img_array = grid.astype(np.uint8) * 255
    # Create image from array
    img = Image.fromarray(img_array, mode='L')
    # Save image
    img.save(output_path)

def display_grid(grid):
    """
    Helper function to display the grid state (useful for debugging)
    
    Args:
        grid (numpy.ndarray): 2D boolean array representing the game state
    """
    for row in grid:
        print(''.join(['█' if cell else '.' for cell in row]))

def run_simulation(initial_grid, num_generations):
    """
    Run the Game of Life simulation for a specified number of generations.
    
    Args:
        initial_grid (numpy.ndarray): 2D boolean array representing the initial state
        num_generations (int): Number of generations to simulate
        
    Returns:
        numpy.ndarray: 2D boolean array representing the final state
    """
    current_grid = initial_grid.copy()
    for _ in range(num_generations):
        current_grid = next_generation(current_grid)
    return current_grid

if __name__ == "__main__":
    # Example usage
    example_path = "../archive/example-0/starting_position.png"
    try:
        # Read initial state
        initial_state = read_starting_position(example_path)
        print("Initial state:")
        display_grid(initial_state)
        
        # Run simulation for different steps and save results
        output_dir = os.path.dirname(example_path)
        steps = [1, 10, 100, 1000]
        
        for step in steps:
            result = run_simulation(initial_state, step)
            output_path = os.path.join(output_dir, f"{step}.png")
            save_grid_as_image(result, output_path)
            print(f"\nAfter {step} generations:")
            display_grid(result)
            
    except Exception as e:
        print(f"Error: {e}") 