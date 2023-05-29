import open3d as o3d
import numpy as np
from time import sleep

try:
    point_array = np.load('temp_p.npy')
    color_array = np.load('temp_c.npy')
    line_point_list = list(np.load('temp_lp.npy'))
    line_line_list = list(np.load('temp_ll.npy'))    
    # check arrays
    show_box = False
    if line_point_list and line_line_list:
        show_box = True


    if show_box:
        line_set = o3d.geometry.LineSet()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array[0,:,:3])
    pcd.colors = o3d.utility.Vector3dVector(color_array[0,:,:3])
    vis.add_geometry(pcd)
    if show_box:
        line_set.points = o3d.utility.Vector3dVector(line_point_list[0])
        line_set.lines = o3d.utility.Vector2iVector(line_line_list[0])

        vis.add_geometry(line_set)


    #vis.update_geometry(pcd)
    if show_box:
        vis.update_geometry(line_set)

    for i in range(1,len(point_array)):
        pcd.points = o3d.utility.Vector3dVector(point_array[i,:,:3])
        pcd.colors = o3d.utility.Vector3dVector(color_array[i,:,:3])

        vis.update_geometry(pcd)

        if show_box:
            line_set.points = o3d.utility.Vector3dVector(line_point_list[i])
            line_set.lines = o3d.utility.Vector2iVector(line_line_list[i])

            vis.update_geometry(line_set)

        vis.poll_events()
        vis.update_renderer()
        sleep(0.00001)

    vis.destroy_window()
except :
    print("Error while reading file")