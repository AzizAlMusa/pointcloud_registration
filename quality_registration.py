import open3d as o3d
import numpy as np
import pandas as pd
import os
import copy

"""
This script performs iterative registration of multiple point clouds using both RANSAC and ICP methods.
It involves filtering point clouds based on alignment scores to improve the quality of the final combined result.

Key Steps:
1. Load point clouds from specified file paths.
2. Preprocess point clouds by filtering, downsampling, and estimating normals and FPFH features.
3. Perform global registration using RANSAC, followed by fine registration using ICP.
4. Filter point clouds based on alignment scores.
5. Combine the filtered and registered point clouds into a final model.
6. Visualize the final combined point cloud.

The script is useful for scenarios where multiple point clouds need to be accurately aligned into a unified model.
"""

def load_point_clouds(file_paths):
    """
    Load multiple point clouds from the given file paths.

    Args:
        file_paths (list of str): List of file paths to the point cloud files.

    Returns:
        list of o3d.geometry.PointCloud: List of loaded point clouds.
    """
    pcds = [o3d.io.read_point_cloud(path) for path in file_paths]
    return pcds

def load_viewpoint(file_path, index):
    """
    Loads a specific viewpoint (transformation matrix) from a CSV file based on an index.

    Args:
        file_path (str): The path to the CSV file containing viewpoints.
        index (int): The index of the viewpoint to load.

    Returns:
        np.ndarray: A 4x4 transformation matrix, or None if the index is out of range.
    """
    df = pd.read_csv(file_path)
    if index < len(df):
        row = df.iloc[index]
        viewpoint = np.eye(4)
        viewpoint[:3, 3] = [row['translation_x'], row['translation_y'], row['translation_z']]
        rotation = o3d.geometry.get_rotation_matrix_from_quaternion([
            row['rotation_w'], row['rotation_x'], row['rotation_y'], row['rotation_z']
        ])
        viewpoint[:3, :3] = rotation
        return viewpoint
    else:
        print("Index out of range for viewpoints.csv")
        return None
    
def create_arrow_from_transform(transform, scale=0.1, color=[0, 0, 1]):
    """
    Creates a 3D arrow mesh to represent a sensor's pose based on a transform matrix.

    Args:
        transform (np.ndarray): The 4x4 transformation matrix.
        scale (float): The scale factor for the arrow size.
        color (list): The RGB color of the arrow.

    Returns:
        o3d.geometry.TriangleMesh: The arrow mesh object.
    """
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.005 * scale,
        cone_radius=0.01 * scale,
        cylinder_height=0.1 * scale,
        cone_height=0.02 * scale
    )
    arrow.transform(transform)
    arrow.paint_uniform_color(color)
    return arrow

def compute_point_to_point_statistics(pcd):
    """
    Computes the mean and standard deviation of nearest neighbor distances in the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud for which to compute statistics.

    Returns:
        tuple: The mean and standard deviation of nearest neighbor distances.
    """
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    distances = []

    for i in range(len(pcd.points)):
        # Find the nearest neighbor distance
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 2)
        nearest_dist = np.linalg.norm(np.asarray(pcd.points[i]) - np.asarray(pcd.points[idx[1]]))
        distances.append(nearest_dist)

    distances = np.array(distances)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    return mean_distance, std_distance

def compute_average_nearest_neighbor_distance(pcd, k=5):
    """
    Computes the average nearest neighbor distance for the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        k (int): The number of nearest neighbors to consider.

    Returns:
        float: The average nearest neighbor distance.
    """
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    distances = []

    for i in range(len(pcd.points)):
        # Find the k nearest neighbors and compute the distance to each
        _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], k + 1)
        distances.extend([np.linalg.norm(np.asarray(pcd.points[i]) - np.asarray(pcd.points[j])) for j in idx[1:]])

    return np.mean(distances)

def estimate_voxel_size(pcd, target_min=10000, target_max=20000, initial_voxel_size=0.01):
    """
    Estimates the voxel size required to downsample a filtered point cloud to achieve a point count 
    between 10,000 and 20,000 points. Points below z = 0.1 are filtered out before estimation.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        target_min (int): The minimum target number of points.
        target_max (int): The maximum target number of points.
        initial_voxel_size (float): The initial voxel size for downsampling.

    Returns:
        float: The estimated voxel size.
    """
    # Filter out points with z < 0.1
    pcd_filtered = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] >= 0.1)[0])
    
    voxel_size = initial_voxel_size
    current_points = len(pcd_filtered.points)
    tolerance = 0.5  # Tolerance value for adjusting voxel size
    
    # Iteratively adjust voxel size until the point count is within the target range
    while True:
        pcd_downsampled = pcd_filtered.voxel_down_sample(voxel_size)
        current_points = len(pcd_downsampled.points)
        
        if target_min <= current_points <= target_max:
            break
        
        if current_points > target_max:
            voxel_size *= (1 + tolerance)
        elif current_points < target_min:
            voxel_size /= (1 + tolerance)

        # Stop if voxel size reaches the bounding box scale
        if voxel_size >= np.linalg.norm(pcd_filtered.get_max_bound() - pcd_filtered.get_min_bound()):
            print("Voxel size reached the scale of the bounding box, further adjustments may not be possible.")
            break
    
    return voxel_size

def adaptive_max_nn(pcd, radius):
    """
    Calculates an adaptive max_nn based on the point cloud density and the chosen radius.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        radius (float): The radius within which to count neighbors.
        
    Returns:
        int: The adaptive max_nn value.
    """
    avg_nn_distance = compute_average_nearest_neighbor_distance(pcd)
    
    # Estimate the number of neighbors within the radius
    if avg_nn_distance > 0:
        expected_nn = int((radius / avg_nn_distance) ** 2)
    else:
        expected_nn = 30  # Fallback value

    return max(30, expected_nn)  # Minimum value for stability

def preprocess_point_cloud(pcd, voxel_size):
    """
    Preprocesses the point cloud by filtering out points below a threshold, 
    removing statistical outliers, downsampling, and estimating normals.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud to preprocess.
        voxel_size (float): The voxel size for downsampling.

    Returns:
        tuple: The downsampled point cloud and its computed FPFH features.
    """
    # Filter points below z < 0.1
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] >= 0.1)[0])
    
    # Remove statistical outliers
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Downsample the point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    bbox = pcd_down.get_axis_aligned_bounding_box()
    bbox_diagonal = np.linalg.norm(bbox.get_extent())
    
    # Adaptive radii based on the bounding box diagonal
    radius_normal = bbox_diagonal * 0.05
    radius_feature = bbox_diagonal * 0.1
    
    # radius_normal = voxel_size * 2
    # radius_feature = voxel_size * 5
    # Adaptive max_nn
    max_nn_normal = adaptive_max_nn(pcd_down, radius_normal)
    max_nn_feature = adaptive_max_nn(pcd_down, radius_feature)
    
    # Estimate normals
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normal))

    # Compute FPFH features
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_feature)
    )
    
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Perform global registration using RANSAC based on feature matching.

    Args:
        source_down (o3d.geometry.PointCloud): The downsampled source point cloud.
        target_down (o3d.geometry.PointCloud): The downsampled target point cloud.
        source_fpfh (o3d.pipelines.registration.Feature): The FPFH feature of the source point cloud.
        target_fpfh (o3d.pipelines.registration.Feature): The FPFH feature of the target point cloud.
        voxel_size (float): The voxel size for determining the distance threshold.

    Returns:
        o3d.pipelines.registration.RegistrationResult: The result of the RANSAC registration.
    """
    # Filter points below z < 0.1 in the source
    source_filtered = source_down.select_by_index(np.where(np.asarray(source_down.points)[:, 2] >= 0.1)[0])
    
    bbox = source_filtered.get_axis_aligned_bounding_box()
    bbox_diagonal = np.linalg.norm(bbox.get_extent())
    
    distance_threshold = bbox_diagonal * 0.01 
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def execute_icp_registration(source, target, transformation, voxel_size):
    """
    Perform fine registration using ICP.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        transformation (np.ndarray): The initial transformation matrix from RANSAC.
        voxel_size (float): The voxel size for determining the distance threshold.

    Returns:
        o3d.pipelines.registration.RegistrationResult: The result of the ICP registration.
    """
    distance_threshold = voxel_size * 0.4
    
    # Estimate normals for both source and target
    radius_normal = voxel_size * 2
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Perform ICP registration
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp

def compute_alignment_scores(pcd, viewpoint):
    """
    Compute alignment scores based on the angle between point normals and the direction of the viewpoint.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud with estimated normals.
        viewpoint (np.ndarray): The 4x4 viewpoint transformation matrix representing both position and orientation.

    Returns:
        np.ndarray: An array of alignment scores for each point in the point cloud.
    """
    # Extract the direction vector from the viewpoint (Z-axis of the rotation matrix)
    direction_vector = viewpoint[:3, 2]  # Z-axis of the rotation part of the transformation matrix
    
    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    
    # Get the normals from the point cloud
    normals = np.asarray(pcd.normals)
    
    # Compute the dot product between the normals and the direction vector
    alignment_scores = np.dot(normals, direction_vector)
    
    # Take the absolute value to ensure the alignment is positive (0 means perpendicular, 1 means perfectly aligned)
    alignment_scores = np.abs(alignment_scores)
    
    return alignment_scores


def create_arrow_from_transform(transform, scale=0.1, color=[0, 0, 1]):
    """
    Creates a 3D arrow mesh to represent a sensor's pose based on a transform matrix.

    Args:
        transform (np.ndarray): The 4x4 transformation matrix.
        scale (float): The scale factor for the arrow size.
        color (list): The RGB color of the arrow.

    Returns:
        o3d.geometry.TriangleMesh: The arrow mesh object.
    """
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.005 * scale,
        cone_radius=0.01 * scale,
        cylinder_height=0.1 * scale,
        cone_height=0.02 * scale
    )
    arrow.transform(transform)
    arrow.paint_uniform_color(color)
    return arrow

def color_points_by_alignment(pcd, alignment_scores):
    """
    Color points in the point cloud based on alignment scores.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to color.
        alignment_scores (np.ndarray): The alignment scores for each point.

    Returns:
        o3d.geometry.PointCloud: The colored point cloud.
    """
    # Normalize alignment scores to be between 0 and 1
    alignment_scores_normalized = (alignment_scores - alignment_scores.min()) / (alignment_scores.max() - alignment_scores.min())
    
    # Create colors based on the normalized alignment scores
    colors = np.zeros((alignment_scores_normalized.shape[0], 3))
    colors[:, 0] = 1 - alignment_scores_normalized  # Red channel decreases with alignment score
    colors[:, 1] = alignment_scores_normalized      # Green channel increases with alignment score
    
    # Assign colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def filter_points_by_alignment(pcd, alignment_scores, threshold=0.75):
    """
    Filter points in the point cloud based on alignment scores, removing poorly aligned points.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to filter.
        alignment_scores (np.ndarray): The alignment scores for each point.
        threshold (float): The minimum alignment score required to keep a point.

    Returns:
        o3d.geometry.PointCloud: The filtered and colored point cloud.
    """
    indices = np.where(alignment_scores >= threshold)[0]
    filtered_pcd = pcd.select_by_index(indices)
    
    # Color the filtered points based on alignment scores
    filtered_pcd = color_points_by_alignment(filtered_pcd, alignment_scores[indices])
    return filtered_pcd


def register(source, target, voxel_size, step):
    """
    Register the source point cloud to the target point cloud, and return the resulting transformation matrix.

    Args:
    source (o3d.geometry.PointCloud): The source point cloud.
    target (o3d.geometry.PointCloud): The target point cloud.
    voxel_size (float): The voxel size for downsampling.
    step (int): The current registration step, used for visualization labeling.

    Returns:
    np.ndarray: The resulting transformation matrix for the current registration.
    """
    # Preprocess both source and target point clouds
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Perform global registration
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(f"RANSAC Registration Result at Step {step}:")
    print(result_ransac)

    # Visualize correspondences after RANSAC
    visualize_correspondences(source_down, target_down, np.asarray(result_ransac.correspondence_set), f"RANSAC Correspondences at Step {step}")

    # Perform ICP registration
    result_icp = execute_icp_registration(source_down, target_down, result_ransac.transformation, voxel_size)
    print(f"ICP Registration Result at Step {step}:")
    print(result_icp)

    # Return the ICP transformation matrix
    return result_icp.transformation

def visualize_correspondences(source, target, correspondences, title="Correspondences"):
    """
    Visualizes the correspondences between the source and target point clouds.

    Args:
    source (o3d.geometry.PointCloud): The source point cloud.
    target (o3d.geometry.PointCloud): The target point cloud.
    correspondences (np.ndarray): The correspondences identified between the source and target.
    title (str): The title of the visualization window.

    Returns:
    None
    """
    # Translate the target point cloud along the Z-axis for clarity
    translation_vector = np.array([0, 0, 0.1])  # Adjust the Z translation as needed
    target_translated = copy.deepcopy(target)
    target_translated.translate(translation_vector)

    # Extract corresponding points from source and translated target
    source_corr = o3d.geometry.PointCloud()
    target_corr = o3d.geometry.PointCloud()

    # Use the correct indices from correspondences to extract points
    source_corr.points = o3d.utility.Vector3dVector(np.asarray(source.points)[correspondences[:, 0]])
    target_corr.points = o3d.utility.Vector3dVector(np.asarray(target_translated.points)[correspondences[:, 1]])

    # Paint the correspondence points with different colors
    source_corr.paint_uniform_color([1, 0, 0])  # Red for source correspondences
    target_corr.paint_uniform_color([0, 0, 1])  # Blue for target correspondences

    # Create lines connecting corresponding points in source and target
    lines = [[i, i + len(source_corr.points)] for i in range(len(correspondences))]
    combined_points = np.vstack((source_corr.points, target_corr.points))
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(combined_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.paint_uniform_color([0, 1, 0])  # Green for correspondences

    # Paint the full downsampled point clouds with different colors
    source.paint_uniform_color([1, 0.6, 0.6])  # Orange for full source
    target_translated.paint_uniform_color([0.6, 0.8, 1.0])  # Light blue for full target

    # Visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title)
    vis.add_geometry(source)
    vis.add_geometry(target_translated)
    vis.add_geometry(source_corr)
    vis.add_geometry(target_corr)
    vis.add_geometry(line_set)
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

def visualize_registration(pcd, title="Registration Result"):
    """
    Visualize the combined point cloud with recomputed normals to restore shading.

    Args:
    pcd (o3d.geometry.PointCloud): The combined point cloud.
    title (str): The title of the visualization window.
    """
    # Recompute normals to restore shading based on lighting
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.0025*2, max_nn=30))
    
    # Assign a uniform gray color to the combined point cloud if needed
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Optional: Gray for combined

    # Create an axis frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Visualize the combined point cloud with the axis frame
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

def main():
    """
    Main function to perform point cloud registration, filtering, and visualization.

    Steps:
    1. Load the point clouds from disk.
    2. Register all point clouds without filtering and save the transformations.
    3. Filter the point clouds based on alignment scores.
    4. Register and combine the filtered point clouds using the saved transformations.
    5. Visualize the final combined result.
    """

    pointcloud_folder = "bell"
    file_paths = [os.path.join(pointcloud_folder, f"{i:03d}.pcd") for i in range(8)]
    viewpoints_file = os.path.join(pointcloud_folder, "viewpoints.csv")
    # Load point clouds
    pcds = load_point_clouds(file_paths)

    # Compute statistics for voxel size estimation
    print("==========================================================")
    print(f"Source has {len(pcds[0].points)} points.")
    print("Computing point-to-point distance statistics on the source point cloud...")
    mean_distance, std_distance = compute_point_to_point_statistics(pcds[0])
    print("Statistics before downsampling:")
    print(f"Mean nearest neighbor distance: {mean_distance:.6f}")
    print(f"Standard deviation of distances: {std_distance:.6f}")
    print("==========================================================")

    # Estimate voxel size for downsampling
    voxel_size = estimate_voxel_size(pcds[0], target_min=10000, target_max=20000, initial_voxel_size=mean_distance)
    print(f"Estimated voxel size: {voxel_size:.6f}")

    # Step 1: Register all point clouds without alignment filtering and save transformations
    registration_transforms = []
    combined = copy.deepcopy(pcds[0])

    for i in range(1, len(pcds)):
        print(f"Registering point cloud {i} with point cloud {i-1}")
        transform = register(pcds[i], pcds[i-1], voxel_size, i)
        registration_transforms.append(transform)

    # Step 2: Filter the alignment for all point clouds
    filtered_pcds = []
    for i, pcd in enumerate(pcds):
        # Use the saved transformation to determine the viewpoint for alignment scoring
        viewpoint = load_viewpoint(viewpoints_file, i)
        
        # Preprocess point cloud
        pcd_down, _ = preprocess_point_cloud(pcd, voxel_size)
        
        # Compute alignment scores based on the viewpoint
        alignment_scores = compute_alignment_scores(pcd_down, viewpoint)
        
        # Filter points based on alignment scores
        filtered_pcd = filter_points_by_alignment(pcd_down, alignment_scores, threshold=0.0)
        filtered_pcds.append(filtered_pcd)

     # Apply the transformations and combine point clouds
    combined = copy.deepcopy(pcds[0])
    for i in range(1, len(pcds)):
        chained_transform = np.eye(4)
        for j in range(i):
            chained_transform = np.dot(chained_transform, registration_transforms[j])
        current_source = copy.deepcopy(pcds[i])
        current_source.transform(chained_transform)
        combined += current_source
    
    combined, ind = combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    
    # Visualize the final combined result
    visualize_registration(combined, "Final Combined Point Cloud")

    # Step 3: Use the saved transformations to register the filtered point clouds together
    # combined_filtered = filtered_pcds[0]
    # for i in range(1, len(filtered_pcds)):
    #     # Apply the saved transformation to align the filtered point clouds
    #     filtered_pcds[i].transform(registration_transforms[i - 1])
    #     combined_filtered += filtered_pcds[i]


    combined_filtered = filtered_pcds[0]
    for i in range(1, len(filtered_pcds)):
        chained_transform = np.eye(4)
        for j in range(i):
            chained_transform = np.dot(chained_transform, registration_transforms[j])
        filtered_pcds[i].transform(chained_transform)
        combined_filtered += filtered_pcds[i]

    # Visualize the final combined result
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([combined_filtered, axis])

if __name__ == "__main__":
    main()
