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

def preprocess_point_cloud(pcd, voxel_size):
    """
    Preprocess the given point cloud by removing points below a certain z value,
    removing outliers, downsampling, and estimating normals and FPFH features.

    Args:
    pcd (o3d.geometry.PointCloud): The input point cloud.
    voxel_size (float): The voxel size for downsampling.

    Returns:
    tuple: The downsampled point cloud and its computed FPFH features.
    """
    # Remove points with z < 0.089
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] >= 0.089)[0])
    
    # Remove statistical outliers
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Downsample the point cloud to reduce computational complexity
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals for the downsampled point cloud
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH features for use in registration
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
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
    distance_threshold = voxel_size * 0.5
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
    
    # Ensure normals are estimated for both source and target point clouds
    radius_normal = voxel_size * 2
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Perform ICP registration to refine the alignment
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp

def compute_alignment_scores(pcd, viewpoint):
    """
    Compute alignment scores based on the angle between point normals and the direction to a viewpoint.

    Args:
    pcd (o3d.geometry.PointCloud): The point cloud with estimated normals.
    viewpoint (np.ndarray): The viewpoint or sensor position.

    Returns:
    np.ndarray: An array of alignment scores for each point in the point cloud.
    """
    normals = np.asarray(pcd.normals)
    directions = np.asarray(pcd.points) - viewpoint
    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    alignment_scores = np.abs(np.sum(normals * directions, axis=1))
    return alignment_scores

def filter_points_by_alignment(pcd, alignment_scores, threshold=0.75):
    """
    Filter points in the point cloud based on alignment scores, removing poorly aligned points.

    Args:
    pcd (o3d.geometry.PointCloud): The point cloud to filter.
    alignment_scores (np.ndarray): The alignment scores for each point.
    threshold (float): The minimum alignment score required to keep a point.

    Returns:
    o3d.geometry.PointCloud: The filtered point cloud.
    """
    indices = np.where(alignment_scores >= threshold)[0]
    filtered_pcd = pcd.select_by_index(indices)
    return filtered_pcd

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
    pointcloud_folder = "pointclouds"
    file_paths = [os.path.join(pointcloud_folder, f"{i:03d}.pcd") for i in range(1, 7)]
    transforms_file = os.path.join(pointcloud_folder, "transforms.csv")

    # Load point clouds
    pcds = load_point_clouds(file_paths)
    voxel_size = 0.0025

    # Step 1: Register all point clouds without alignment filtering and save transformations
    transforms = []
    target = pcds[0]
    combined = copy.deepcopy(target)

    for i in range(1, len(pcds)):
        source = pcds[i]
        
        # Preprocess both source and target
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

        # Perform global registration using RANSAC
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        
        # Perform ICP registration to refine the transformation
        result_icp = execute_icp_registration(source, target, result_ransac.transformation, voxel_size)
        
        # Save the transformation found by ICP for later use
        transforms.append(result_icp.transformation)

        # Update the target for the next iteration (apply transformation to source)
        target = copy.deepcopy(source)
        target.transform(result_icp.transformation)

    # Step 2: Filter the alignment for all point clouds
    filtered_pcds = []
    for i, pcd in enumerate(pcds):
        # Use the saved transformation to determine the viewpoint for alignment scoring
        transform = transforms[i - 1] if i > 0 else np.eye(4)
        viewpoint = transform[:3, 3]
        
        # Preprocess point cloud
        pcd_down, _ = preprocess_point_cloud(pcd, voxel_size)
        
        # Compute alignment scores based on the viewpoint
        alignment_scores = compute_alignment_scores(pcd_down, viewpoint)
        
        # Filter points based on alignment scores
        filtered_pcd = filter_points_by_alignment(pcd_down, alignment_scores, threshold=0.65)
        filtered_pcds.append(filtered_pcd)

    # Step 3: Use the saved transformations to register the filtered point clouds together
    combined = filtered_pcds[0]
    for i in range(1, len(filtered_pcds)):
        # Apply the saved transformation to align the filtered point clouds
        filtered_pcds[i].transform(transforms[i - 1])
        combined += filtered_pcds[i]

    # Visualize the final combined result
    combined.paint_uniform_color([0.5, 0.5, 0.5])  # Optional: Apply a uniform gray color
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # Add an axis frame for reference
    o3d.visualization.draw_geometries([combined, axis])

if __name__ == "__main__":
    main()
