import open3d as o3d
import numpy as np
import copy

"""
This script performs 3D point cloud registration using both RANSAC and ICP methods and visualizes the results.
It includes functionality to load, preprocess, and analyze point clouds, as well as to compute registration
transformations and visualize both the raw and aligned point clouds. The script also provides the ability to 
visualize correspondences identified during the registration process.

Key functionalities:
- Load source and target point clouds from files.
- Visualize the original, downsampled, and aligned point clouds.
- Preprocess the point clouds by removing outliers, downsampling, and estimating normals.
- Compute nearest neighbor distance statistics for analysis.
- Perform global registration using RANSAC, followed by fine registration using ICP.
- Visualize the correspondences between source and target points.
- Optionally skip RANSAC and directly perform ICP registration with an identity transformation.


"""

import open3d as o3d
import numpy as np
import copy

def load_point_clouds(source_path, target_path):
    """
    Loads the source and target point clouds from the specified file paths.

    Args:
        source_path (str): The file path to the source point cloud.
        target_path (str): The file path to the target point cloud.

    Returns:
        tuple: The loaded source and target point clouds.
    """
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    return source, target

def visualize_point_clouds(source, target, title="Original Point Clouds"):
    """
    Visualizes the source and target point clouds, coloring them differently.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        title (str): The title of the visualization window.

    Returns:
        None
    """
    # Paint the point clouds with different colors
    source.paint_uniform_color([1, 0, 0])  # Red for source
    target.paint_uniform_color([0, 0, 1])  # Blue for target

    # Visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title)
    vis.add_geometry(source)
    vis.add_geometry(target)
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

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
    # Remove points with z < 0.089
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] >= 0.089)[0])
    
    # Remove outliers using a statistical method
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Downsample the point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Compute the bounding box and its volume for analysis
    # bbox = pcd_down.get_axis_aligned_bounding_box()
    # volume = np.prod(bbox.get_extent())
    # print(f"Bounding box volume of downsampled point cloud: {volume:.6f}")

    # Estimate normals for the downsampled point cloud
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features for the downsampled point cloud
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

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
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], 2)
        nearest_dist = np.linalg.norm(np.asarray(pcd.points[i]) - np.asarray(pcd.points[idx[1]]))
        distances.append(nearest_dist)

    distances = np.array(distances)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    return mean_distance, std_distance

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Performs global registration using RANSAC based on feature matching.

    Args:
        source_down (o3d.geometry.PointCloud): The downsampled source point cloud.
        target_down (o3d.geometry.PointCloud): The downsampled target point cloud.
        source_fpfh (o3d.pipelines.registration.Feature): The FPFH features of the source point cloud.
        target_fpfh (o3d.pipelines.registration.Feature): The FPFH features of the target point cloud.
        voxel_size (float): The voxel size used for determining thresholds.

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
    Refines the initial registration using the Iterative Closest Point (ICP) method.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        transformation (np.ndarray): The initial transformation matrix.
        voxel_size (float): The voxel size used for determining the distance threshold.

    Returns:
        o3d.pipelines.registration.RegistrationResult: The result of the ICP registration.
    """
    distance_threshold = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp

def visualize_registration(source, target, transformation, title):
    """
    Visualizes the result of the registration process by showing the source, 
    target, and transformed source point clouds.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        transformation (np.ndarray): The transformation matrix applied to the source point cloud.
        title (str): The title of the visualization window.

    Returns:
        None
    """
    # Transform the source point cloud with the provided transformation
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    
    # Paint the point clouds with different colors
    source.paint_uniform_color([1, 0, 0])  # Red for original source
    source_transformed.paint_uniform_color([0, 1, 0])  # Green for transformed source
    target.paint_uniform_color([0, 0, 1])  # Blue for target
    
    # Create an axis frame for reference
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title)
    vis.add_geometry(source)
    vis.add_geometry(source_transformed)
    vis.add_geometry(target)
    vis.add_geometry(axis)

    vis.update_renderer()
    vis.run()
    vis.destroy_window()

def visualize_correspondences(source, target, correspondences):
    """
    Visualizes the correspondences between the source and target point clouds.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        correspondences (np.ndarray): The correspondences identified between the source and target.

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
    vis.create_window(window_name="Correspondences and Point Clouds")
    vis.add_geometry(source)
    vis.add_geometry(target_translated)
    vis.add_geometry(source_corr)
    vis.add_geometry(target_corr)
    vis.add_geometry(line_set)
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

def main(skip_ransac=False):
    """
    Main function to execute the point cloud registration workflow.
    It loads point clouds, preprocesses them, performs registration using RANSAC and ICP,
    and visualizes the results at various stages.

    Args:
        skip_ransac (bool): Whether to skip the RANSAC registration step and directly use ICP.

    Returns:
        None
    """
    # Load point clouds
    source_path = "data/005.pcd"
    target_path = "data/006.pcd"
    source, target = load_point_clouds(source_path, target_path)

    # Visualize the original point clouds before preprocessing
    visualize_point_clouds(source, target, title="Original Point Clouds")
    
    # Compute point-to-point distance statistics of source
    print(f"Source has {len(source.points)} points.")
    mean_distance, std_distance = compute_point_to_point_statistics(source)
    print("Statistics before downsampling:")
    print(f"Mean nearest neighbor distance: {mean_distance:.6f}")
    print(f"Standard deviation of distances: {std_distance:.6f}")

    # Preprocess point clouds
    voxel_size = 0.0025
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Compute point-to-point distance statistics of downsampled source
    print(f"Source downsampled has {len(source_down.points)} points.")
    mean_distance, std_distance = compute_point_to_point_statistics(source_down)
    print("Statistics after downsampling:")
    print(f"Mean nearest neighbor distance: {mean_distance:.6f}")
    print(f"Standard deviation of distances: {std_distance:.6f}")

    if not skip_ransac:
        # Perform global registration using RANSAC
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        print("RANSAC Registration Result:")
        print(result_ransac)

        # Visualize correspondences chosen by RANSAC
        visualize_correspondences(source_down, target_down, np.asarray(result_ransac.correspondence_set))

        # Visualize pre-ICP registration (after RANSAC)
        visualize_registration(source, target, result_ransac.transformation, title="Pre-ICP Registration")
        
        # Use RANSAC transformation as initial for ICP
        initial_transformation = result_ransac.transformation
    else:
        # Skip RANSAC and use identity transformation as initial for ICP
        print("Skipping RANSAC, using identity transformation for ICP.")
        initial_transformation = np.identity(4)

    # Perform ICP registration
    result_icp = execute_icp_registration(source_down, target_down, initial_transformation, voxel_size)
    print("ICP Registration Result:")
    print(result_icp)

    # Visualize post-ICP registration
    visualize_registration(source, target, result_icp.transformation, title="Post-ICP Registration")

if __name__ == "__main__":
    # Set skip_ransac to True if you want to skip RANSAC and only use ICP
    main(skip_ransac=False)
