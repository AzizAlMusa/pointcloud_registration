import open3d as o3d
import numpy as np
import pandas as pd
import os

"""
This script processes and visualizes a point cloud from a set of point clouds stored in a directory. 
It includes various functions to filter the point cloud based on z-values, downsample the point cloud, 
estimate normals, and compute alignment scores between the point cloud's normals and a specified viewpoint. 
The script also loads transformation matrices from a CSV file to visualize the sensor's pose with respect 
to the point cloud. 

The main functionality includes:
- Loading a point cloud file.
- Filtering points based on a minimum z-value.
- Downsampling the point cloud to reduce data density.
- Estimating normals for the downsampled point cloud.
- Computing alignment scores between the normals and a viewpoint extracted from a transform matrix.
- Filtering and coloring the point cloud based on these alignment scores.
- Visualizing the point cloud alongside an arrow representing the sensor pose.


"""

import open3d as o3d
import numpy as np
import pandas as pd
import os

def load_point_cloud(file_path):
    """
    Loads a point cloud from a specified file.

    Args:
        file_path (str): The path to the point cloud file.

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud object.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def filter_by_z_value(pcd, min_z=0.09):
    """
    Filters the point cloud to remove points below a certain z-value.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to filter.
        min_z (float): The minimum z-value threshold.

    Returns:
        o3d.geometry.PointCloud: The filtered point cloud.
    """
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] >= min_z)[0])
    return pcd

def downsample_point_cloud(pcd, voxel_size):
    """
    Downsamples the point cloud using a voxel grid filter.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to downsample.
        voxel_size (float): The size of the voxel grid.

    Returns:
        o3d.geometry.PointCloud: The downsampled point cloud.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down

def estimate_normals(pcd, radius):
    """
    Estimates normals for the point cloud using a KD-tree search.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud for which to estimate normals.
        radius (float): The radius for the KD-tree search parameter.

    Returns:
        o3d.geometry.PointCloud: The point cloud with estimated normals.
    """
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return pcd

def load_transform(file_path, index):
    """
    Loads a specific transformation matrix from a CSV file based on an index.

    Args:
        file_path (str): The path to the CSV file containing transforms.
        index (int): The index of the transform to load.

    Returns:
        np.ndarray: A 4x4 transformation matrix, or None if the index is out of range.
    """
    df = pd.read_csv(file_path)
    if index < len(df):
        row = df.iloc[index]
        transform = np.eye(4)
        transform[:3, 3] = [row['translation_x'], row['translation_y'], row['translation_z']]
        rotation = o3d.geometry.get_rotation_matrix_from_quaternion([
            row['rotation_w'], row['rotation_x'], row['rotation_y'], row['rotation_z']
        ])
        transform[:3, :3] = rotation
        return transform
    else:
        print("Index out of range for transforms.csv")
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

def compute_alignment_scores(pcd, viewpoint):
    """
    Computes the alignment scores between the point cloud normals and a viewpoint.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud with normals.
        viewpoint (np.ndarray): The viewpoint to compare against.

    Returns:
        np.ndarray: An array of alignment scores.
    """
    normals = np.asarray(pcd.normals)
    directions = np.asarray(pcd.points) - viewpoint
    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    # Compute alignment (dot product and take absolute value)
    alignment_scores = np.abs(np.sum(normals * directions, axis=1))
    return alignment_scores

def color_points_by_alignment(pcd, scores):
    """
    Colors the points of a point cloud based on alignment scores.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to color.
        scores (np.ndarray): The alignment scores to base the coloring on.

    Returns:
        None
    """
    colors = np.zeros((len(scores), 3))
    colors[:, 0] = 1.0 - scores  # Red channel decreases as score increases
    colors[:, 1] = scores  # Green channel increases as score increases
    pcd.colors = o3d.utility.Vector3dVector(colors)

def filter_points_by_alignment(pcd, alignment_scores, threshold=0.75):
    """
    Filters points from a point cloud based on alignment scores and a threshold.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to filter.
        alignment_scores (np.ndarray): The alignment scores for the points.
        threshold (float): The minimum alignment score threshold.

    Returns:
        o3d.geometry.PointCloud: The filtered point cloud.
    """
    indices = np.where(alignment_scores >= threshold)[0]
    filtered_pcd = pcd.select_by_index(indices)
    return filtered_pcd

def visualize_with_arrow(pcd, arrow):
    """
    Visualizes the point cloud with an arrow representing the sensor's pose.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to visualize.
        arrow (o3d.geometry.TriangleMesh): The arrow representing the sensor's pose.

    Returns:
        None
    """
    o3d.visualization.draw_geometries([pcd, arrow], point_show_normal=False)

def main():
    """
    Main function to execute the point cloud processing and visualization pipeline.
    """
    # Load point cloud
    pointcloud_folder = "pointclouds"
    file_name = "003.pcd"  # Replace with your point cloud file name
    file_path = os.path.join(pointcloud_folder, file_name)
    pcd = load_point_cloud(file_path)
    
    # Filter points below z = 0.09
    pcd = filter_by_z_value(pcd, min_z=0.092)
    
    # Downsample point cloud
    voxel_size = 0.0025  # Adjust voxel size as needed
    pcd_down = downsample_point_cloud(pcd, voxel_size)
    
    # Estimate normals after downsampling
    radius = voxel_size * 2  # Adjust radius based on the scale of your point cloud
    pcd_down = estimate_normals(pcd_down, radius)
    
    # Determine the index based on the point cloud file name (e.g., 003.pcd -> index 3)
    file_index = int(os.path.splitext(file_name)[0])
    
    # Load the specific transform for this point cloud
    transforms_file = os.path.join(pointcloud_folder, "transforms.csv")
    transform = load_transform(transforms_file, file_index)
    
    if transform is not None:
        # Extract the viewpoint from the transform
        viewpoint = transform[:3, 3]
        
        # Compute alignment scores
        alignment_scores = compute_alignment_scores(pcd_down, viewpoint)
        
        # Filter points by alignment score threshold
        threshold = 0.0
        filtered_pcd = filter_points_by_alignment(pcd_down, alignment_scores, threshold)
        
        # Color the filtered points based on alignment scores
        color_points_by_alignment(filtered_pcd, alignment_scores[alignment_scores >= threshold])
        
        # Create an arrow to represent the transform
        arrow = create_arrow_from_transform(transform, scale=0.4, color=[0, 0, 1])
        
        # Visualize the filtered and colored point cloud with the arrow
        visualize_with_arrow(filtered_pcd, arrow)
    else:
        print("No matching transform found for the given point cloud.")

if __name__ == "__main__":
    main()
