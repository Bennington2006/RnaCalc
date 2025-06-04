import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import trimesh
import igl
from scipy.spatial import cKDTree
from numba import njit
# Function to extract RNA spots from text file
def extract_rna_spots(file):
    rna_data = pd.read_csv(file, sep='\t')
    print(rna_data.columns)  # Print the column names to inspect them
    #rna_spots = rna_data[['Position X', 'Position Y', 'Position Z']].values
    rna_spots = rna_data.iloc[:, :3].values
    rna_spots = np.round(rna_spots, 4)
    return rna_spots


# Function to filter spots inside the mesh
def filter_spots_inside_mesh(mesh, spots): ##update this using winding number
    spots = spots[np.all(np.isfinite(spots), axis=1)]
    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds
    
    # Filter spots to be within the bounding box
    within_bounds = np.all((spots >= min_bound) & (spots <= max_bound), axis=1)
    spots_within_bounds = spots[within_bounds]
    
    # Check if the remaining spots are inside the mesh
    inside = mesh.contains(spots_within_bounds)
    
    return spots_within_bounds[inside]

def colocper(spots1, spots2, threshold=1):
    # Calculate pairwise Euclidean distances
    distances = cdist(spots1, spots2)
    # Count the number of spots in spots1 within the threshold distance of spots2
    close_spots1 = np.any(distances <= threshold, axis=1)
    percentage_spots1 = np.mean(close_spots1) * 100
    # Count the number of spots in spots2 within the threshold distance of spots1
    close_spots2 = np.any(distances <= threshold, axis=0)
    percentage_spots2 = np.mean(close_spots2) * 100
    result_dict = {
        'coloc_1_with_2': percentage_spots1,
        'coloc_2_with_1': percentage_spots2
    }
    return result_dict



def colocper_three(spots1, spots2, spots3, threshold=1):
    # Find colocalization between spots1 and spots2
    distances_12 = cdist(spots1, spots2)
    coloc_12 = np.any(distances_12 <= threshold, axis=1)
    coloc_spots1 = spots1[coloc_12]

    # Find colocalization of the previously colocalized spots with spots3
    if coloc_spots1.size == 0:
        return {"coloc_all": 0.0} 
    
    distances_123 = cdist(coloc_spots1, spots3)
    coloc_123 = np.any(distances_123 <= threshold, axis=1)

    # Calculate percentage
    percentage_coloc = np.mean(coloc_123) * 100
    return {"coloc_all": percentage_coloc}



def colocper_three_e(spots1, spots2, spots3, threshold=1):
    # Distances
    d12 = cdist(spots1, spots2)
    d13 = cdist(spots1, spots3)

    # Boolean masks for colocalization
    coloc_12 = np.any(d12 <= threshold, axis=1)
    coloc_13 = np.any(d13 <= threshold, axis=1)

    # Compute categories
    coloc_both = coloc_12 & coloc_13
    coloc_2_only = coloc_12 & ~coloc_13
    coloc_3_only = coloc_13 & ~coloc_12

    total = len(spots1)
    return {
        "coloc_both": 100 * np.sum(coloc_both) / total,
        "coloc_2_only": 100 * np.sum(coloc_2_only) / total,
        "coloc_3_only": 100 * np.sum(coloc_3_only) / total
    }


def generate_random_spots(mesh, desired_points, oversample_factor=2):
    # Get the bounding box
    min_bound, max_bound = mesh.bounds

    points_inside_mesh = []

    while len(points_inside_mesh) < desired_points:
        # Sample more points
        num_points_to_generate = (desired_points - len(points_inside_mesh))
        random_points = np.random.uniform(low=min_bound, high=max_bound, size=(num_points_to_generate, 3))
        inside = mesh.contains(random_points)
        points_inside_mesh.extend(random_points[inside])
    #convert to np array
    points_inside_mesh = np.array(points_inside_mesh)
    
    return points_inside_mesh

def rotate_rna_spots(rna_spots, vertices_np):
    # Calculate the center of the mesh
    mesh_center = vertices_np.mean(axis=0)

    # Translate the rna_spots to the origin
    rna_spots_centered = rna_spots - mesh_center

    # Define the rotation matrix for a 90-degree clockwise rotation around the z-axis
    rotation_matrix_z_clockwise = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    # Apply the rotation matrix
    rna_spots_rotated = np.dot(rna_spots_centered, rotation_matrix_z_clockwise.T)

    # Translate the rna_spots back to the original center
    rna_spots_rotated = rna_spots_rotated + mesh_center

    return rna_spots_rotated


def calculate_max_density(rna_spots, mesh, resolution=1.0):
    # Get the bounding box
    min_bound, max_bound = mesh.bounds

    # Calculate the number of bins along each axis
    num_bins = np.ceil((max_bound - min_bound) / resolution).astype(int)

    # Initialize a 3D grid to count the number of spots in each cubic micrometer
    density_grid = np.zeros(num_bins, dtype=int)

    # Shift the spots so that the bounding box starts at the origin
    shifted_spots = rna_spots - min_bound

    # Calculate the bin indices for each spot
    bin_indices = np.floor(shifted_spots / resolution).astype(int)

    # Count the number of spots in each bin
    for idx in bin_indices:
        if np.all(idx >= 0) and np.all(idx < num_bins):
            density_grid[tuple(idx)] += 1

    # Calculate the density for each cubic micrometer
    density_grid = density_grid / (resolution ** 3)

    # Find the maximum density
    max_density = np.max(density_grid)

    return max_density



def generate_kde_spots3(rna_spots, mesh):
    # Estimate KDE for rna_spots
    kde_model = gaussian_kde(rna_spots.T)
   
    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds
    bbox_volume = np.prod(max_bound - min_bound)

    spots_per_cubi_um = calculate_max_density(rna_spots, mesh)

    desired_points = int(bbox_volume * spots_per_cubi_um)
    sample_space = np.random.uniform(low=mesh.bounds[0], high=mesh.bounds[1], size=(desired_points, 3))

    V = np.array(mesh.vertices)
    F = np.array(mesh.faces)
    # Compute winding number for the sample space
    W = igl.fast_winding_number_for_meshes(V, F, sample_space)
    # Keep only the points where winding number >= 0.5 (i.e., inside the mesh)
    inside_mask = W >= 0.5
    points_inside_mesh1 = sample_space[inside_mask]

    #change to np array
    points_inside_mesh = np.array(points_inside_mesh1)

    # Evaluate KDE on the sampled points
    kde_eval = kde_model(points_inside_mesh.T)

    # Normalize KDE values to create probability weights
    probabilities = kde_eval / np.sum(kde_eval)

    # Sample points based on KDE probability distribution
    sampled_indices = np.random.choice(len(points_inside_mesh), len(rna_spots), replace=True, p=probabilities)
    KDE_spots = points_inside_mesh[sampled_indices]

    return KDE_spots

#for images
def generate_kde_spots4(rna_spots, mesh):
    # Estimate KDE for rna_spots
    kde_model = gaussian_kde(rna_spots.T)
   
    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds
    bbox_volume = np.prod(max_bound - min_bound)

    spots_per_cubi_um = calculate_max_density(rna_spots, mesh)

    desired_points = int(bbox_volume * spots_per_cubi_um)
    sample_space = np.random.uniform(low=mesh.bounds[0], high=mesh.bounds[1], size=(desired_points, 3))

    V = np.array(mesh.vertices)
    F = np.array(mesh.faces)
    # Compute winding number for the sample space
    W = igl.fast_winding_number_for_meshes(V, F, sample_space)
    # Keep only the points where winding number >= 0.5 (i.e., inside the mesh)
    inside_mask = W >= 0.5
    points_inside_mesh1 = sample_space[inside_mask]

    #change to np array
    points_inside_mesh = np.array(points_inside_mesh1)

    # Evaluate KDE on the sampled points
    kde_eval = kde_model(points_inside_mesh.T)

    # Normalize KDE values to create probability weights
    probabilities = kde_eval / np.sum(kde_eval)

    # Sample points based on KDE probability distribution
    sampled_indices = np.random.choice(len(points_inside_mesh), len(rna_spots), replace=True, p=probabilities)
    KDE_spots = points_inside_mesh[sampled_indices]

    return KDE_spots, points_inside_mesh1, probabilities

# Define functions for random trials and KDE trials
def run_random_trial(args):
    import pandas as pd  # Ensure pandas is imported within the function
    mesh, rna_spots_1, rna_spots_2, coloc_threshold = args
    try:
        random_rna_spots_1 = generate_random_spots(mesh, desired_points=len(rna_spots_1))
        random_rna_spots_2 = generate_random_spots(mesh, desired_points=len(rna_spots_2))
        return colocper(random_rna_spots_1, random_rna_spots_2, threshold=coloc_threshold)
    except Exception as e:
        print(f"Error in run_random_trial: {e}")
        return None

def run_random_trial_triple(args):
    import pandas as pd  # Ensure pandas is imported within the function
    
    # Unpack the arguments
    mesh, rna_spots_1, rna_spots_2, rna_spots_3, coloc_threshold = args
    try:
        # Generate random spots for each RNA channel with the same count as the original data
        random_rna_spots_1 = generate_random_spots(mesh, desired_points=len(rna_spots_1))
        random_rna_spots_2 = generate_random_spots(mesh, desired_points=len(rna_spots_2))
        random_rna_spots_3 = generate_random_spots(mesh, desired_points=len(rna_spots_3))
        
        # Perform three-channel colocalization analysis
        return colocper_three(random_rna_spots_1, random_rna_spots_2, random_rna_spots_3, threshold=coloc_threshold)
    except Exception as e:
        print(f"Error in run_random_trial: {e}")
        return None

def run_random_trial_triple_e(args):
    import pandas as pd  # Ensure pandas is imported within the function
    
    # Unpack the arguments
    mesh, rna_spots_1, rna_spots_2, rna_spots_3, coloc_threshold = args
    try:
        # Generate random spots for each RNA channel with the same count as the original data
        random_rna_spots_1 = generate_random_spots(mesh, desired_points=len(rna_spots_1))
        random_rna_spots_2 = generate_random_spots(mesh, desired_points=len(rna_spots_2))
        random_rna_spots_3 = generate_random_spots(mesh, desired_points=len(rna_spots_3))
        
        # Perform three-channel colocalization analysis
        return colocper_three_e(random_rna_spots_1, random_rna_spots_2, random_rna_spots_3, threshold=coloc_threshold)
    except Exception as e:
        print(f"Error in run_random_trial: {e}")
        return None        

def run_kde_trial(args):
    import pandas as pd  # Ensure pandas is imported within the function
    mesh, rna_spots_1, rna_spots_2, coloc_threshold= args
    try:
           # Calculate the oversample multiplier
        kde_randomized_rna_spots_1 = generate_kde_spots3(rna_spots_1, mesh)
        kde_randomized_rna_spots_2 = generate_kde_spots3(rna_spots_2, mesh)
        return colocper(kde_randomized_rna_spots_1, kde_randomized_rna_spots_2, threshold=coloc_threshold)
    except Exception as e:
        print(f"Error in run_kde_trial: {e}")
        return None




def run_kde_trial_triple(args):
    import pandas as pd  # Ensure pandas is imported within the function
    mesh, rna_spots_1, rna_spots_2, rna_spots_3, coloc_threshold = args
    try:
        kde_randomized_rna_spots_1 = generate_kde_spots3(rna_spots_1, mesh)
        kde_randomized_rna_spots_2 = generate_kde_spots3(rna_spots_2, mesh)
        kde_randomized_rna_spots_3 = generate_kde_spots3(rna_spots_3, mesh)
        return colocper_three(
            kde_randomized_rna_spots_1, 
            kde_randomized_rna_spots_2, 
            kde_randomized_rna_spots_3, 
            threshold=coloc_threshold
        )
    except Exception as e:
        print(f"Error in run_kde_trial: {e}")
        return None


def run_kde_trial_triple_e(args):
    import pandas as pd  # Ensure pandas is imported within the function
    mesh, rna_spots_1, rna_spots_2, rna_spots_3, coloc_threshold = args
    try:
        kde_randomized_rna_spots_1 = generate_kde_spots3(rna_spots_1, mesh)
        kde_randomized_rna_spots_2 = generate_kde_spots3(rna_spots_2, mesh)
        kde_randomized_rna_spots_3 = generate_kde_spots3(rna_spots_3, mesh)
        return colocper_three_e(
            kde_randomized_rna_spots_1, 
            kde_randomized_rna_spots_2, 
            kde_randomized_rna_spots_3, 
            threshold=coloc_threshold
        )
    except Exception as e:
        print(f"Error in run_kde_trial: {e}")
        return None

def compute_original_colocalization(rna_spots_list, num_trials, coloc_threshold):
    """
    Compute the original colocalization percentage between RNA spot sets.

    Args:
        rna_spots_list (list of np.ndarray): List of RNA spot coordinate arrays.
        num_trials (int): Number of trials (not used here but kept for consistency).
        coloc_threshold (float): Distance threshold for colocalization.

    Returns:
        float: Average colocalization percentage.
    """
    if len(rna_spots_list) < 2:
        raise ValueError("At least two RNA spot sets are required for colocalization analysis.")

    # Compute pairwise colocalization percentages
    coloc_results = []
    for i in range(len(rna_spots_list) - 1):
        for j in range(i + 1, len(rna_spots_list)):
            coloc_result = colocper(rna_spots_list[i], rna_spots_list[j], threshold=coloc_threshold)
            coloc_results.append(coloc_result['coloc_1_with_2'])

    # Return the average colocalization percentage
    return np.mean(coloc_results)


def compute_rotated_colocalization(rna_spots_list, mesh, num_trials, coloc_threshold):
    """
    Compute colocalization after rotating RNA spots.
    Args:
        rna_spots_list (list of np.ndarray): List of RNA spot coordinate arrays.
        mesh (trimesh.Trimesh): The mesh object representing the cell.
        num_trials (int): Number of trials (not used here but kept for consistency).
        coloc_threshold (float): Distance threshold for colocalization.
    Returns:
        float: Average colocalization percentage after rotation.
    """
    rotated_results = []
    for rna_spots in rna_spots_list:
        rotated_spots = rotate_rna_spots(rna_spots, np.array(mesh.vertices))
        coloc_result = colocper(rotated_spots, rna_spots_list[0], threshold=coloc_threshold)
        rotated_results.append(coloc_result['coloc_1_with_2'])
    return np.mean(rotated_results)


def compute_random_colocalization(rna_spots_list, mesh, num_trials, coloc_threshold):
    """
    Compute colocalization using randomized RNA spots.
    Args:
        rna_spots_list (list of np.ndarray): List of RNA spot coordinate arrays.
        mesh (trimesh.Trimesh): The mesh object representing the cell.
        num_trials (int): Number of random trials.
        coloc_threshold (float): Distance threshold for colocalization.
    Returns:
        float: Average colocalization percentage from random trials.
    """
    random_results = []
    for _ in range(num_trials):
        random_spots_list = [generate_random_spots(mesh, len(rna_spots)) for rna_spots in rna_spots_list]
        coloc_result = colocper(random_spots_list[0], random_spots_list[1], threshold=coloc_threshold)
        random_results.append(coloc_result['coloc_1_with_2'])
    return np.mean(random_results)


def compute_kde_colocalization(rna_spots_list, mesh, num_trials, coloc_threshold):
    """
    Compute colocalization using KDE-based RNA spots.
    Args:
        rna_spots_list (list of np.ndarray): List of RNA spot coordinate arrays.
        mesh (trimesh.Trimesh): The mesh object representing the cell.
        num_trials (int): Number of KDE trials.
        coloc_threshold (float): Distance threshold for colocalization.
    Returns:
        float: Average colocalization percentage from KDE trials.
    """
    kde_results = []
    for _ in range(num_trials):
        kde_spots_list = [generate_kde_spots3(rna_spots, mesh) for rna_spots in rna_spots_list]
        coloc_result = colocper(kde_spots_list[0], kde_spots_list[1], threshold=coloc_threshold)
        kde_results.append(coloc_result['coloc_1_with_2'])
    return np.mean(kde_results)