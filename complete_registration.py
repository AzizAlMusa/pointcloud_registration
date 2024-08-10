import open3d as o3d
import numpy as np
import copy

"""
This script implements an iterative 3D point cloud registration and visualization process using Open3D.
The primary goal of the script is to align multiple point clouds into a single, unified point cloud representation.

The script performs the following steps:

1. **Loading Point Clouds:**
   - Multiple point clouds are loaded from specified file paths and stored in a list.

2. **Preprocessing:**
   - Each point cloud undergoes several preprocessing steps:
     - Points below a specified z-value are filtered out.
     - Statistical outliers are removed to clean up noise.
     - The point cloud is downsampled using voxel-based sampling to reduce computational load.
     - Normals are estimated for the downsampled point cloud.
     - Fast Point Feature Histograms (FPFH) features are computed for use in feature-based registration.

3. **Global Registration (RANSAC):**
   - A global registration is performed between each pair of point clouds using the RANSAC algorithm.
   - This step aligns the point clouds based on their FPFH features, providing an initial rough alignment.

4. **Fine Registration (ICP):**
   - The rough alignment from RANSAC is refined using the Iterative Closest Point (ICP) algorithm.
   - ICP minimizes the point-to-plane distances between the source and target point clouds for better accuracy.

5. **Registration and Combination:**
   - The source point cloud is transformed based on the ICP result and then combined with the target point cloud.
   - This process is repeated iteratively for each subsequent point cloud, progressively building up a combined point cloud.

6. **Visualization:**
   - The script visualizes the correspondences between point clouds after both RANSAC and ICP steps.
   - The combined point cloud is visualized with recomputed normals to restore shading effects.
   - An optional filtered version of the combined point cloud, removing points below a certain z-value, is also visualized.

7. **Final Output:**
   - The script outputs a final visualization of the combined point cloud after all registrations are completed.
   - The final visualization helps assess the quality of the registration and alignment process.

This script is particularly useful for scenarios where multiple point clouds, representing different perspectives of the same object or scene, need to be aligned into a cohesive model.
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
    o3d.geometry.PointCloud: The downsampled point cloud.
    o3d.pipelines.registration.Feature: The FPFH feature of the downsampled point cloud.
    """
    # Remove points with z < 0.089
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] >= 0.089)[0])
    
    # Remove outliers
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Downsample the point cloud
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH feature
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Perform global registration using RANSAC.

    Args:
    source_down (o3d.geometry.PointCloud): The downsampled source point cloud.
    target_down (o3d.geometry.PointCloud): The downsampled target point cloud.
    source_fpfh (o3d.pipelines.registration.Feature): The FPFH feature of the source point cloud.
    target_fpfh (o3d.pipelines.registration.Feature): The FPFH feature of the target point cloud.
    voxel_size (float): The voxel size for downsampling.

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
    transformation (np.ndarray): The initial transformation matrix.
    voxel_size (float): The voxel size for downsampling.

    Returns:
    o3d.pipelines.registration.RegistrationResult: The result of the ICP registration.
    """
    distance_threshold = voxel_size * 0.4

    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp

def register_and_combine(source, target, voxel_size, step):
    """
    Register the source point cloud to the target point cloud and combine them,
    then visualize the correspondences and combined point cloud.

    Args:
    source (o3d.geometry.PointCloud): The source point cloud.
    target (o3d.geometry.PointCloud): The target point cloud.
    voxel_size (float): The voxel size for downsampling.
    step (int): The current registration step, used for visualization labeling.

    Returns:
    o3d.geometry.PointCloud: The combined point cloud.
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

    # Transform the source point cloud and combine it with the target point cloud
    source.transform(result_icp.transformation)
    combined = target + source

    # Visualize the combined result
    visualize_registration(combined, f"Combined Point Cloud at Step {step}")

    return combined


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



def visualize_filtered_registration(pcd):
    """
    Visualize the combined point cloud with points below 0.089 in the z-direction removed and colored dark green.

    Args:
    pcd (o3d.geometry.PointCloud): The combined point cloud.
    """
    # Remove points with z < 0.089
    filtered_pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] >= 0.089)[0])
    
    # Assign a dark green color to the filtered point cloud
    filtered_pcd.paint_uniform_color([0.0, 0.5, 0.0])  # Dark green

    # Create an axis frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Visualize the filtered point cloud with the axis frame
    o3d.visualization.draw_geometries([filtered_pcd, axis])

def main():
    """
    Main function to load, register, combine, and visualize point clouds.
    """
    # Define the file paths for the point clouds
    file_paths = [f"data/00{i}.pcd" for i in range(1, 7)]
    
    # Load the point clouds
    pcds = load_point_clouds(file_paths)
    
    # Define the voxel size for downsampling
    voxel_size = 0.0025

    # Initialize the aggregate point cloud as the first point cloud
    combined = copy.deepcopy(pcds[0])

    # Register each point cloud with the current aggregate
    for i in range(1, len(pcds)):
        combined = register_and_combine(pcds[i], combined, voxel_size, i)

    # Visualize the final combined result
    visualize_registration(combined, "Final Combined Point Cloud")

    # Visualize the filtered result
    visualize_filtered_registration(combined)


if __name__ == "__main__":
    main()
