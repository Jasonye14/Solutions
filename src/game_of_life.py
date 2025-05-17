import numpy as np
from PIL import Image
import os
from collections import defaultdict

# Global flag to ensure detailed (1,6) trace happens only for the first relevant call
debug_trace_done_for_1_6_gen0 = False

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

def resolve_neighbor_coordinates(r_cell, c_cell, dr_query, dc_query, h_portals, v_portals, grid_rows, grid_cols):
    """
    Resolves the effective coordinates of a neighbor for a given cell and direction,
    considering wormhole rules and precedence (Top > Right > Bottom > Left).

    Args:
        r_cell (int): Row of the current cell.
        c_cell (int): Column of the current cell.
        dr_query (int): Row offset for the queried neighbor direction (-1, 0, or 1).
        dc_query (int): Column offset for the queried neighbor direction (-1, 0, or 1).
        h_portals (dict): Horizontal wormhole portal map.
        v_portals (dict): Vertical wormhole portal map.
        grid_rows (int): Total rows in the grid.
        grid_cols (int): Total columns in the grid.

    Returns:
        tuple: (eff_r, eff_c) effective coordinates of the neighbor.
    """

    # Check Vertical Wormhole influence (Top/Bottom precedence)
    v_pair_coords = v_portals.get((r_cell, c_cell))
    if v_pair_coords:
        v_pair_r, v_pair_c = v_pair_coords
        # Case 1: (r_cell,c_cell) is TOP end of V-wormhole, looking DOWN (dr_query=1)
        if dr_query == 1 and r_cell < v_pair_r:
            return v_pair_r, v_pair_c + dc_query
        # Case 2: (r_cell,c_cell) is BOTTOM end of V-wormhole, looking UP (dr_query=-1)
        if dr_query == -1 and r_cell > v_pair_r:
            return v_pair_r, v_pair_c + dc_query

    # Check Horizontal Wormhole influence (Right/Left precedence)
    h_pair_coords = h_portals.get((r_cell, c_cell))
    if h_pair_coords:
        h_pair_r, h_pair_c = h_pair_coords
        # Case 1: (r_cell,c_cell) is LEFT end of H-wormhole, looking RIGHT (dc_query=1)
        if dc_query == 1 and c_cell < h_pair_c:
            return h_pair_r + dr_query, h_pair_c
        # Case 2: (r_cell,c_cell) is RIGHT end of H-wormhole, looking LEFT (dc_query=-1)
        if dc_query == -1 and c_cell > h_pair_c:
            return h_pair_r + dr_query, h_pair_c

    # Default: No overriding wormhole rule applied for this (cell, direction) combination
    return r_cell + dr_query, c_cell + dc_query

def count_neighbors(grid, h_portals, v_portals, generation_count_for_debug=0):
    """
    Count the number of live neighbors for each cell in the grid,
    considering wormholes.
    
    Args:
        grid (numpy.ndarray): 2D boolean array representing the current state.
        h_portals (dict): Horizontal wormhole portal map.
        v_portals (dict): Vertical wormhole portal map.
        generation_count_for_debug (int): Passed to control one-time debug print.
        
    Returns:
        numpy.ndarray: 2D integer array with the count of live neighbors for each cell.
    """
    global debug_trace_done_for_1_6_gen0
    rows, cols = grid.shape
    neighbor_counts = np.zeros((rows, cols), dtype=int)
    
    # Determine if this is the specific call we want to trace (first generation calculation)
    is_first_gen_count = (generation_count_for_debug == 0)

    for r in range(rows):
        for c in range(cols):
            should_trace_this_cell = is_first_gen_count and (r == 1 and c == 6) and not debug_trace_done_for_1_6_gen0
            
            if should_trace_this_cell:
                print(f"\n--- Counting neighbors for cell ({r},{c}) (State: {'Live' if grid[r,c] else 'Dead'}) during Gen 0->1 calculation ---")

            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    eff_r, eff_c = resolve_neighbor_coordinates(r, c, dr, dc, h_portals, v_portals, rows, cols)
                    
                    is_live_neighbor = False
                    if 0 <= eff_r < rows and 0 <= eff_c < cols:
                        if grid[eff_r, eff_c]:
                            is_live_neighbor = True
                            count += 1
                    
                    if should_trace_this_cell:
                        original_neighbor = (r + dr, c + dc)
                        print(f"  Query dir ({dr},{dc}): Original N {original_neighbor} -> Effective N ({eff_r},{eff_c}). Effective N is Live: {is_live_neighbor}")
            
            neighbor_counts[r, c] = count
            if should_trace_this_cell:
                print(f"--- Total live neighbors for ({r},{c}): {count} ---")
                debug_trace_done_for_1_6_gen0 = True # Mark trace as done for this specific scenario
            
    return neighbor_counts

def next_generation(grid, h_portals, v_portals, generation_count_for_debug=0):
    """
    Compute the next generation of the Game of Life based on the current state.
    
    Rules:
    1. Any live cell with fewer than two live neighbors dies (underpopulation)
    2. Any live cell with two or three live neighbors lives on
    3. Any live cell with more than three live neighbors dies (overpopulation)
    4. Any dead cell with exactly three live neighbors becomes alive (reproduction)
    
    Args:
        grid (numpy.ndarray): 2D boolean array representing the current state
        h_portals (dict): Horizontal wormhole portal map.
        v_portals (dict): Vertical wormhole portal map.
        generation_count_for_debug (int): Passed to control one-time debug print in count_neighbors.
        
    Returns:
        numpy.ndarray: 2D boolean array representing the next generation
    """
    print(f"DEBUG: next_generation called for generation_count_for_debug = {generation_count_for_debug}") # Debug print
    neighbor_counts = count_neighbors(grid, h_portals, v_portals, generation_count_for_debug)
    
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

def run_simulation(initial_grid, num_generations, h_portals, v_portals):
    """
    Run the Game of Life simulation for a specified number of generations.
    
    Args:
        initial_grid (numpy.ndarray): 2D boolean array representing the initial state
        num_generations (int): Number of generations to simulate
        h_portals (dict): Horizontal wormhole portal map.
        v_portals (dict): Vertical wormhole portal map.
        
    Returns:
        numpy.ndarray: 2D boolean array representing the final state
    """
    current_grid = initial_grid.copy()
    for gen_idx in range(num_generations):
        current_grid = next_generation(current_grid, h_portals, v_portals, gen_idx)
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

def read_wormhole_map(image_path):
    """
    Reads a wormhole map image and returns a dictionary mapping portal coordinates to their pairs.
    Non-black pixels of the same color define a wormhole pair.

    Args:
        image_path (str): Path to the wormhole tunnel image file.

    Returns:
        dict: A dictionary where keys are (row, col) tuples of a portal cell,
              and values are (row, col) tuples of the paired portal cell.
              Returns an empty dict if file not found or no portals defined.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Wormhole map file not found: {image_path}")
        return {}

    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB')
        width, height = img_rgb.size
        
        color_to_coords = defaultdict(list)
        for x in range(width):
            for y in range(height):
                color = img_rgb.getpixel((x, y))
                if color != (0, 0, 0): # Not black
                    color_to_coords[color].append((y, x)) # Store as (row, col)
        
        portals = {}
        for color, coords_list in color_to_coords.items():
            if len(coords_list) == 2:
                p1, p2 = coords_list[0], coords_list[1]
                portals[p1] = p2
                portals[p2] = p1
            else:
                print(f"Warning: Color {color} in {image_path} does not define a pair (found {len(coords_list)} pixels).")
        return portals

if __name__ == "__main__":
    # Reset the global flag at the start of each run if necessary, though for a single run this is fine.
    debug_trace_done_for_1_6_gen0 = False

    # Example usage
    example_dir = "../archive/example-0"
    start_pos_file = "starting_position.png"
    h_tunnel_file = "horizontal_tunnel.png"
    v_tunnel_file = "vertical_tunnel.png"

    example_path = os.path.join(example_dir, start_pos_file)
    h_tunnel_path = os.path.join(example_dir, h_tunnel_file)
    v_tunnel_path = os.path.join(example_dir, v_tunnel_file)

    try:
        # Read initial state
        initial_state = read_starting_position(example_path)
        print("Initial state:")
        display_grid(initial_state)

        # Read wormhole maps
        print(f"Loading horizontal wormholes from: {h_tunnel_path}")
        h_portals = read_wormhole_map(h_tunnel_path)
        print(f"Found {len(h_portals)//2 if h_portals else 0} horizontal wormhole pairs.")
        if h_portals:
            print("Horizontal Portals (first 5 pairs):")
            for i, (p_start, p_end) in enumerate(h_portals.items()):
                if i < 10 and p_start < p_end: # Print each pair once, max 5 pairs
                     print(f"  {p_start} <-> {p_end}")
                if i >= 10: break # limit printing
        
        print(f"Loading vertical wormholes from: {v_tunnel_path}")
        v_portals = read_wormhole_map(v_tunnel_path)
        print(f"Found {len(v_portals)//2 if v_portals else 0} vertical wormhole pairs.")
        if v_portals:
            print("Vertical Portals (first 5 pairs):")
            for i, (p_start, p_end) in enumerate(v_portals.items()):
                if i < 10 and p_start < p_end: # Print each pair once, max 5 pairs
                    print(f"  {p_start} <-> {p_end}")
                if i >= 10: break # limit printing

        # Run simulation for different steps and save results
        output_dir = os.path.dirname(example_path)
        steps = [1]
        
        for step in steps:
            result = run_simulation(initial_state, step, h_portals, v_portals)
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