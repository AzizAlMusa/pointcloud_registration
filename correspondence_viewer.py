import open3d as o3d
import numpy as np
import copy

"""
This script performs 3D point cloud registration and visualizes the correspondences between two point clouds. 
The workflow includes loading the point clouds, preprocessing them by filtering and downsampling, performing 
global registration using RANSAC, and then visualizing the found correspondences interactively.

Key functionalities:
- Loading source and target point clouds from files.
- Preprocessing point clouds by removing outliers, downsampling, and estimating normals.
- Computing FPFH features for use in registration.
- Performing global registration using RANSAC to find correspondences between the two point clouds.
- Visualizing the correspondences interactively with the ability to cycle through each correspondence.

Controls:
- Press 'N' to move to the next correspondence.
- Press 'P' to move to the previous correspondence.

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

def preprocess_point_cloud(pcd, voxel_size):
    """
    Preprocesses the point cloud by filtering, downsampling, and estimating normals.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): The voxel size for downsampling.

    Returns:
        tuple: The downsampled point cloud and its corresponding FPFH features.
    """
    # Remove points with z < 0.089 to filter out ground or irrelevant data
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] >= 0.089)[0])
    
    # Remove statistical outliers to clean the point cloud
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Downsample the point cloud to reduce the number of points
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals based on the downsampled point cloud
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH features for the downsampled point cloud
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Executes global registration between the source and target point clouds using RANSAC.

    Args:
        source_down (o3d.geometry.PointCloud): The downsampled source point cloud.
        target_down (o3d.geometry.PointCloud): The downsampled target point cloud.
        source_fpfh (o3d.pipelines.registration.Feature): The FPFH features of the source point cloud.
        target_fpfh (o3d.pipelines.registration.Feature): The FPFH features of the target point cloud.
        voxel_size (float): The voxel size used in the registration process.

    Returns:
        o3d.pipelines.registration.RegistrationResult: The result of the RANSAC registration process.
    """
    distance_threshold = voxel_size * 1.5 # Define the distance threshold for RANSAC
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

def get_correspondences(result_ransac):
    """
    Extracts correspondences from the RANSAC registration result.

    Args:
        result_ransac (o3d.pipelines.registration.RegistrationResult): The RANSAC registration result.

    Returns:
        list: A list of tuples representing correspondences between source and target points.
    """
    correspondences = []
    for corr in result_ransac.correspondence_set:
        correspondences.append((corr[0], corr[1]))
    return correspondences

class CorrespondenceVisualizer:
    """
    A class to visualize the correspondences between source and target point clouds.
    The visualizer allows iterating through correspondences and visualizing them one by one.
    """
    
    def __init__(self, source, target, correspondences):
        """
        Initializes the visualizer with the source and target point clouds and their correspondences.

        Args:
            source (o3d.geometry.PointCloud): The source point cloud.
            target (o3d.geometry.PointCloud): The target point cloud.
            correspondences (list): The list of correspondences between the source and target points.
        """
        self.source = source
        self.target = target
        self.correspondences = correspondences
        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.init_camera_view()
        self.update_visualization()
        
        # Register keyboard callbacks to move between correspondences
        self.vis.register_key_callback(ord("N"), self.next_correspondence)
        self.vis.register_key_callback(ord("P"), self.prev_correspondence)

    def compute_centroid(self):
        """
        Computes the centroid of the combined point cloud (source + target).

        Returns:
            np.ndarray: The centroid of the combined point cloud.
        """
        source_points = np.asarray(self.source.points)
        target_points = np.asarray(self.target.points)
        all_points = np.vstack((source_points, target_points))
        centroid = np.mean(all_points, axis=0)
        return centroid

    def init_camera_view(self):
        """
        Initializes the camera view to focus on the centroid of the combined point cloud.
        """
        centroid = self.compute_centroid()
        ctr = self.vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        params.extrinsic = np.array([[1, 0, 0, -centroid[0]],
                                     [0, 1, 0, -centroid[1]],
                                     [0, 0, 1, -centroid[2]],
                                     [0, 0, 0, 1]])
        ctr.convert_from_pinhole_camera_parameters(params)

    def update_visualization(self):
        """
        Updates the visualization with the current correspondence.
        Displays the source and target point clouds, and highlights the current correspondence with spheres and a line.
        """
        self.vis.clear_geometries()

        # Save current camera parameters
        ctr = self.vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()

        source_copy = copy.deepcopy(self.source)
        target_copy = copy.deepcopy(self.target)

        # Paint point clouds light transparent gray for better visibility
        source_copy.paint_uniform_color([0.5, 0.5, 0.5])
        target_copy.paint_uniform_color([0.5, 0.5, 0.5])
        source_copy.translate((0, 0, 0)) # Translate to avoid overlap for better visualization
        target_copy.translate((0.0, 0.0, 0.1))

        # Extract the current correspondence
        correspondence = self.correspondences[self.index]
        source_point = source_copy.points[correspondence[0]]
        target_point = target_copy.points[correspondence[1]]

        # Create spheres for the corresponding points
        sphere_source = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        sphere_source.paint_uniform_color([0.5, 0.0, 0.0])  # Dark Red for source
        sphere_source.translate(source_point)

        sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        sphere_target.paint_uniform_color([0.5, 0.0, 0.0])  # Dark Red for target
        sphere_target.translate(target_point)

        # Create a line between the corresponding points
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([source_point, target_point]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line_set.colors = o3d.utility.Vector3dVector([[0.5, 0.0, 0.0]])  # Dark Red line

        # Add geometries to the visualizer
        self.vis.add_geometry(source_copy)
        self.vis.add_geometry(target_copy)
        self.vis.add_geometry(sphere_source)
        self.vis.add_geometry(sphere_target)
        self.vis.add_geometry(line_set)
        
        # Restore camera parameters
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        
        # Update the visualizer
        self.vis.poll_events()
        self.vis.update_renderer()

    def next_correspondence(self, vis):
        """
        Moves to the next correspondence in the list and updates the visualization.
        """
        self.index = (self.index + 1) % len(self.correspondences)
        self.update_visualization()

    def prev_correspondence(self, vis):
        """
        Moves to the previous correspondence in the list and updates the visualization.
        """
        self.index = (self.index - 1) % len(self.correspondences)
        self.update_visualization()

    def run(self):
        """
        Runs the visualizer, allowing interactive exploration of correspondences.
        """
        self.vis.run()
        self.vis.destroy_window()

def main():
    """
    Main function to load point clouds, perform preprocessing and registration, 
    and visualize the correspondences interactively.
    """
    # Load source and target point clouds from files
    source_path = "data/003.pcd"
    target_path = "data/004.pcd"
    source, target = load_point_clouds(source_path, target_path)

    # Preprocess the point clouds (filter, downsample, estimate normals, compute FPFH features)
    voxel_size = 0.005
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Perform global registration using RANSAC
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)

    # Extract and print correspondences
    correspondences = get_correspondences(result_ransac)
    print(f"Number of correspondences found: {len(correspondences)}")
   

    # Visualize the preprocessed source and target with iterating correspondences
    visualizer = CorrespondenceVisualizer(source_down, target_down, correspondences)
    visualizer.run()

if __name__ == "__main__":
    main()
