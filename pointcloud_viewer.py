import open3d as o3d
import os
import numpy as np
import matplotlib.cm as cm

"""
This script is designed to visualize the point cloud files using Open3D. 
The selected point clouds are downsampled and colored based on a Plasma colormap 
to differentiate them visually. The script processes the specified point clouds 
from a designated data folder, applies uniform color to each point cloud, and 
renders them together in a single visualization window.

Instructions:
- Place your point cloud files (.pcd) in the 'data' folder.
- Specify the filenames you wish to visualize in the 'subset_files' list.
- Optionally adjust the voxel size in the downsampling step to control the resolution.
- Run the script to visualize the selected point clouds.
"""

# Directory containing the point cloud files
data_folder = "bell"

# Specify the subset of files you want to visualize
# subset_files = ['000.pcd','001.pcd','002.pcd','003.pcd', '004.pcd','005.pcd',  '006.pcd', '007.pcd']  # Example subset
subset_files = ['004.pcd', '005.pcd']  # Example subset

# Generate colors using the Plasma colormap
plasma = cm.get_cmap('plasma', len(subset_files))
colors = [plasma(i)[:3] for i in range(len(subset_files))]  # Get RGB values

# List to hold all point clouds
point_clouds = []

# Downsample and color each point cloud in the subset
for idx, pcd_file in enumerate(subset_files):
    file_path = os.path.join(data_folder, pcd_file)
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Downsample the point cloud
    # pcd = pcd.voxel_down_sample(voxel_size=0.02)  # Adjust voxel size as needed
    
    # Assign a Plasma color to the point cloud
    color = colors[idx]
    pcd.paint_uniform_color(color)
    
    # Append the colored point cloud to the list
    point_clouds.append(pcd)

# Visualize all selected point clouds in one window
o3d.visualization.draw_geometries(point_clouds)

print("Selected point clouds have been visualized.")
