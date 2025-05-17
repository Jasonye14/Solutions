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

def display_grids_side_by_side(grid1, grid2, label1="Simulated", label2="Expected"):
    """
    Display two grids side by side for comparison
    
    Args:
        grid1 (numpy.ndarray): First grid to display
        grid2 (numpy.ndarray): Second grid to display
        label1 (str): Label for the first grid
        label2 (str): Label for the second grid
    """
    # Calculate the width of each grid for proper spacing
    width = len(grid1[0])
    # Print headers
    print(f"\n{label1}{' ' * (width - len(label1))} | {label2}")
    print("-" * width + "+" + "-" * (width + 1))
    
    # Print rows side by side
    for row1, row2 in zip(grid1, grid2):
        line1 = ''.join(['█' if cell else '.' for cell in row1])
        line2 = ''.join(['█' if cell else '.' for cell in row2])
        print(f"{line1} | {line2}")

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

def compare_grids(grid1, grid2):
    """
    Compare two grids and return whether they match and the percentage of matching cells.
    
    Args:
        grid1 (numpy.ndarray): First grid to compare
        grid2 (numpy.ndarray): Second grid to compare
        
    Returns:
        tuple: (bool, float) - Whether grids match exactly and percentage of matching cells
    """
    if grid1.shape != grid2.shape:
        return False, 0.0
    
    matches = np.sum(grid1 == grid2)
    total_cells = grid1.size
    match_percentage = (matches / total_cells) * 100
    return np.array_equal(grid1, grid2), match_percentage

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
            
            # Compare with expected result
            expected_path = os.path.join(output_dir, f"expected-{step}.png")
            if os.path.exists(expected_path):
                expected_state = read_starting_position(expected_path)
                exact_match, match_percentage = compare_grids(result, expected_state)
                display_grids_side_by_side(result, expected_state)
                print(f"\nComparison with expected result:")
                print(f"Exact match: {'Yes' if exact_match else 'No'}")
                print(f"Matching cells: {match_percentage:.2f}%")
            else:
                print(f"No expected result file found for {step} generations")
                display_grid(result)
            
    except Exception as e:
        print(f"Error: {e}") 