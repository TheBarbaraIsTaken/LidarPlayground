import rosbag
import os
import numpy as np
from sensor_msgs.point_cloud2 import read_points
import velodyne_decoder as vd
import open3d as o3d
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from pcap_processer import PcapReader, ProcessUnit
from collections import deque
from enum import Enum

class State(Enum):
    EMPTY = 0
    DEFAULT = 1
    HEATMAP = 2
    GROUND_FILTER = 3


class LidarProcesser:
    def __init__(self):
        self.cloud_arrays_filtered = []
        self.cloud_arrays_heatmap = []
        self.cloud_arrays = []
        self.color_arrays = []
        self.color_theme = [1, 0.706, 0]
        self.line_line_list = []
        self.line_point_list = []
        self.__show_box = False
        self.cluster_num = 1
        self.box_num = 0
        self.time = 0
        self.__status = State.EMPTY
        LidarProcesser.ValidExtensions = ('.pcap', '.pcd', '.bag')

    def new_file(self, file_path):
        if not file_path:
            return
        
        self.file_path = file_path
        self.cloud_arrays_heatmap = []
        self.cloud_arrays_filtered = []
        self.cloud_arrays = []
        self.color_arrays = []
        self.cluster_num = 1
        self.box_num = 0
        self.time = 0
        self.__show_box = False
        self.__status = State.DEFAULT
        

        _, ext = os.path.splitext(self.file_path)

        if ext == ".bag":
            self.__read_bag()
        elif ext == ".pcap":
            self.__read_pcap()
        elif ext == ".pcd":
            self.__read_pcd()

        if not self.cloud_arrays:
            self.__status = State.EMPTY

    def save(self, file_name, i):
        if self.__status == State.EMPTY:
            return

        if self.__status == State.DEFAULT:
            arr = self.cloud_arrays
        elif self.__status == State.HEATMAP:
            arr = self.cloud_arrays_heatmap
        else:
            arr = self.cloud_arrays_filtered

        if i >= len(arr):
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr[i][:,:3])
        pcd.colors = o3d.utility.Vector3dVector(self.color_arrays[i][:,:3])

        o3d.io.write_point_cloud(file_name, pcd)

    def is_pcap(self):
        if not self.file_path:
            return False
        _, ext = os.path.splitext(self.file_path)

        if ext != '.pcap':
            return False
        
        return True

    def is_empty(self):
        return self.__status == State.EMPTY
    
    def is_show_box(self):
        return self.__show_box
    
    def set_show_box(self, val):
        self.__show_box = val
    
    def set_status(self, status):
        if self.is_empty():
            return
        self.__status = status

    def get_status(self):
        return self.__status
    
    def render_image(self, point_cloud):
        if self.is_empty():
            return np.full((480, 640, 4), 255, dtype=np.uint8)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], s=1, c=self.get_colors()[0])

        # Set plot limits and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])

        # Vary the viewpoint
        ax.view_init(elev=30, azim=60)

        # Save plot as numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        image = np.array(canvas.renderer.buffer_rgba())

        return image
    
    @staticmethod
    def list2array(arr):
        if not list(arr):
            return []
        max_len = max(map(len, arr))
        new_shape = (len(arr), max_len, len(arr[0][0]))
        new_arr = np.full(new_shape, np.nan)

        for i, pcd in enumerate(arr):
            new_arr[i,:len(pcd)] = pcd

        return new_arr
    
    def get_color_array(self, lower, upper):
        return LidarProcesser.list2array(self.color_arrays)

    def get_array(self, lower, upper):
        if self.__status == State.EMPTY:
            return []
        elif self.__status == State.GROUND_FILTER:
            return LidarProcesser.list2array(self.cloud_arrays_filtered[lower:upper])
        elif self.__status == State.HEATMAP:
            return self.cloud_arrays_heatmap
        
        return LidarProcesser.list2array(self.cloud_arrays[lower:upper])
    
    def get_cluster_time(self):
        return round(self.time, 2)
    
    def get_extents(self):
        if self.is_empty():
            return 0,0,0

        arr = self.cloud_arrays
        if self.__status == State.GROUND_FILTER:
            arr = self.cloud_arrays_filtered

        max_x = max(map(lambda xyz: np.nanmax(xyz[:,0]), arr))
        min_x = min(map(lambda xyz: np.nanmin(xyz[:,0]), arr))

        max_y = max(map(lambda xyz: np.nanmax(xyz[:,1]), arr))
        min_y = min(map(lambda xyz: np.nanmin(xyz[:,1]), arr))

        max_z = max(map(lambda xyz: np.nanmax(xyz[:,2]), arr))
        min_z = min(map(lambda xyz: np.nanmin(xyz[:,2]), arr))

        return int(max_x-min_x), int(max_y-min_y), int(max_z-min_z)

    def get_cluster_num(self):
        return round(self.cluster_num, 2)

    def get_lineset_l(self):
        if not self.line_line_list or self.is_empty():
            return []
        return LidarProcesser.list2array(self.line_line_list)
    
    def get_lineset_p(self):
        if not self.line_point_list or self.is_empty():
            return []
        return LidarProcesser.list2array(self.line_point_list)
    
    def get_colors(self):
        return self.color_arrays
    
    def get_frame_num(self):
        return len(self.cloud_arrays)
    
    def get_avg_pcd_len(self):
        if self.is_empty():
            return 0

        arr = self.cloud_arrays
        if self.__status == State.GROUND_FILTER:
            arr = self.cloud_arrays_filtered

        return round(sum(map(lambda pcd: len(pcd), arr)) / len(arr), 2)
    
    def get_box_num(self):
        return round(self.box_num, 2)
    
    def get_first_frame(self):
        if self.__status == State.EMPTY:
            return []
        elif self.__status == State.GROUND_FILTER:
            return self.cloud_arrays_filtered[0]
        elif self.__status == State.HEATMAP:
            if len(self.cloud_arrays_heatmap) > 1:
                return self.cloud_arrays_heatmap[1]
            return self.cloud_arrays_heatmap[0]

        return self.cloud_arrays[0]
    
    def none_colors(self, lower, upper):
        # set attributes to default
        self.time = 0
        self.cluster_num = 1
        self.box_num = 0
        self.line_point_list = []
        self.line_line_list = []
        self.color_arrays = []

        if self.__status == State.GROUND_FILTER:
            arr = self.cloud_arrays_filtered
        else:
            arr = self.cloud_arrays
        
        for points in arr[lower:upper]:
            self.color_arrays.append(np.broadcast_to(self.color_theme, (len(points), 3)))

    def __ransac(self, iter, treshold, xyz):
        max_z = max(map(lambda xyz: np.nanmax(xyz[:,2]), self.cloud_arrays))
        max_z = 0
        min_z = min(map(lambda xyz: np.nanmin(xyz[:,2]), self.cloud_arrays))
        
        best_z = 0
        best_fit = -1
        best_mask = False
        for i in range(iter):
            z = random.uniform(min_z, max_z)
            mask = (xyz[:,2] < z + treshold) & (xyz[:,2] > z - treshold)
            fit = sum(mask)

            if fit > best_fit:
                best_fit = fit
                best_z = z
                best_mask = mask
        
        return best_z, best_mask
    
    def filter_ground(self, lower, upper, iter, treshold):
        self.cloud_arrays_filtered = []
        for points in self.cloud_arrays[lower:upper]:
            _, mask = self.__ransac(iter, treshold, points)
            self.cloud_arrays_filtered.append(points[~mask])
    
    @staticmethod
    def is_valid_box(b_points, valid_extents):
        # width, length, height: x, y, z
        max_w, min_w, max_h, min_h, max_l, min_l = valid_extents

        max_x = np.nanmax(b_points[:,0])
        min_x = np.nanmin(b_points[:,0])
        w = max_x-min_x
        if w > max_w or w < min_w:
            return False

        max_y = np.nanmax(b_points[:,1])
        min_y = np.nanmin(b_points[:,1])
        l = max_y-min_y
        if l > max_l or w < min_l:
            return False

        max_z = np.nanmax(b_points[:,2])
        min_z = np.nanmin(b_points[:,2])
        w = max_z-min_z
        if w > max_h or w < min_h:
            return False
        
        return True
    
    @staticmethod
    def filter_nans(arr):
        new_list = []
        for item in arr:
            new_list.append(item[~np.isnan(item[:,0])])
        return new_list
    
    def dbscan_colors(self, eps, min_samples, lower, upper, extents):
        self.cluster_num = 0
        self.box_num = 0
        self.color_arrays = []
        cloud_arrays = []
        self.line_point_list = []
        self.line_line_list = []
        line_order = np.asarray([[0,1],[0,2],[1,7],[2,7],[3,5],[3,6],[4,5],[4,6],[0,3],[2,5],[1,6],[4,7]])

        if self.__status == State.GROUND_FILTER:
            arr = self.cloud_arrays_filtered
        else:
            arr = self.cloud_arrays

        arr = LidarProcesser.filter_nans(arr)

        pcd = o3d.geometry.PointCloud()
        start = time.time()
        for points in arr[lower:upper]:
            line_points = []
            line_lines = []
            i = 0

            pcd.points = o3d.utility.Vector3dVector(points)
            down_pcd = pcd.voxel_down_sample(0.05)
            cloud_arrays.append(np.asarray(down_pcd.points))

            labels = np.array(down_pcd.cluster_dbscan(eps, min_samples))
            max_label = np.nanmax(labels)
            self.cluster_num += (max_label+1)
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            
            self.color_arrays.append(colors)

            if self.__show_box:
                for label in range(0,max_label+1):
                    indices = list(map(lambda x: x[0], filter(lambda x: x[1] == label, (enumerate(labels)))))
                    points_array = np.take(down_pcd.points, indices, 0)

                    points_vector = o3d.utility.Vector3dVector(np.ndarray(points_array.shape, buffer=np.array(points_array)))
                    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points_vector)
                    box_points = bbox.get_box_points()

                    if LidarProcesser.is_valid_box(np.asarray(box_points), extents):
                        for p in box_points:
                            line_points.append(p)
                            
                        for l in line_order+(8*i):
                            line_lines.append(l)
                        
                        i += 1

            if self.__show_box:
                self.box_num += i
                self.line_point_list.append(line_points)
                self.line_line_list.append(line_lines)

        end = time.time()

        if self.__status == State.GROUND_FILTER:
            self.cloud_arrays_filtered = cloud_arrays
        else:
            self.cloud_arrays = cloud_arrays

        self.cluster_num /= (upper-lower)
        self.box_num /= (upper-lower)
        self.time = end-start
    
    def kmeans_colors(self, n, lower, upper, extents):
        self.cluster_num = n
        self.box_num = 0
        self.line_point_list = []
        self.line_line_list = []
        line_order = np.asarray([[0,1],[0,2],[1,7],[2,7],[3,5],[3,6],[4,5],[4,6],[0,3],[2,5],[1,6],[4,7]])

        kmeans = KMeans(
            init="random",
            n_clusters=n,
            n_init=10,
            max_iter=300,
            random_state=42
        )

        cloud_arrays = []
        self.color_arrays = []

        calc_label = lambda p: np.argmin(np.sum((kmeans.cluster_centers_- p)**2, axis=1))
        pcd = o3d.geometry.PointCloud()

        if self.__status == State.GROUND_FILTER:
            arr = self.cloud_arrays_filtered
        else:
            arr = self.cloud_arrays

        arr = LidarProcesser.filter_nans(arr)

        start = time.time()
        for points in arr[lower:upper]:
            pcd.points = o3d.utility.Vector3dVector(points)
            down_pcd = pcd.voxel_down_sample(0.05)
            points = np.asarray(down_pcd.points)
            cloud_arrays.append(np.asarray(points))
            line_points = []
            line_lines = []
            i = 0

            kmeans.fit(down_pcd.points)
            labels = np.apply_along_axis(calc_label, axis=1, arr=points)
            max_label = np.nanmax(labels)
            colors = np.asarray(plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1)))
            self.color_arrays.append(colors)

            # Bounding boxes
            if self.__show_box:
                for label in range(0,max_label+1):
                    indices = list(map(lambda x: x[0], filter(lambda x: x[1] == label, (enumerate(labels)))))
                    points_array = np.take(points, indices, 0)

                    points_vector = o3d.utility.Vector3dVector(np.ndarray(points_array.shape, buffer=np.array(points_array)))
                    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points_vector)

                    box_points = bbox.get_box_points()

                    if LidarProcesser.is_valid_box(np.asarray(box_points), extents):
                        for p in box_points:
                            line_points.append(p)
                            
                        for l in line_order+(8*i):
                            line_lines.append(l)
                        
                        i += 1

            if self.__show_box:
                self.box_num += i
                self.line_point_list.append(line_points)
                self.line_line_list.append(line_lines)

        end = time.time()

        if self.__status == State.GROUND_FILTER:
            self.cloud_arrays_filtered = cloud_arrays
        else:
            self.cloud_arrays = cloud_arrays

        self.box_num /= (upper-lower)
        self.time = end-start
        
    def __read_bag(self):
        try:
            self.cloud_arrays = []
            bag = rosbag.Bag(self.file_path)
            for _, msg, _ in bag.read_messages(topics=['velodyne_points']):
                point_list = list(read_points(msg))
                velodyne_points = np.array(point_list, dtype=np.float32)
                self.color_arrays.append(np.broadcast_to(self.color_theme, (len(point_list), 3)))
                self.cloud_arrays.append(velodyne_points[:,:3])
            bag.close()
        except Exception:
            self.cloud_arrays = []

    def __read_pcap(self):
        try:
            self.cloud_arrays = []
            config = vd.Config(model='VLP-16', rpm=600)
            for _, points in vd.read_pcap(self.file_path, config):
                points = np.asarray(points)
                self.color_arrays.append(np.broadcast_to(self.color_theme, (len(points), 3)))
                self.cloud_arrays.append(points[:,:3])
        except Exception:
            self.cloud_arrays = []

    def __read_pcd(self):
        try:
            self.cloud_arrays = []
            pcd = o3d.io.read_point_cloud(self.file_path)
            points = (np.asarray(pcd.points))
            if (list(points)):
                self.cloud_arrays = [np.asarray(pcd.points)]
            colors = np.asarray(pcd.colors)
            if list(colors):
                self.color_arrays.append(colors)
            else:
                self.color_arrays.append(np.broadcast_to(self.color_theme, (len(self.cloud_arrays[0]), 3)))
        except Exception:
            self.cloud_arrays = []

    def __calc_ttc(self, package_num, prc):
        reader = PcapReader(self.file_path)
        reader.read_file()
    
        delta_t = 0.1
        weights = np.asarray([1,2,3,4,5,6,7,8,9,8,8,7,7,6,4,2])
        # from 15 degree to -15 degree
        order = [15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0]

        window_size = 20
        max_pcd_len = 907 # maximum number of blocks in a pcd
        min_pcd_len = 340

        ttc_list = []

        azimuth_prev = 223
        prev_package = 0
        inx = 0
        pcd_length = 0
        distance_shape = (package_num,32,4)
        distance_packages = np.full(distance_shape, np.nan)
        differences = deque(maxlen=window_size)

        distance_list = []

        for block in reader.blocks:
            azimuth = prc.calc_azimuth(block)

            if not prc.is_valid_azimuth(azimuth, azimuth_prev):
                continue

            block_values = prc.calc_block_values(block)
            current_package = prc.calc_package_index(azimuth)
            if prev_package != current_package:
                inx = 0

            distance_packages[current_package,:len(block_values),inx] = block_values
    
            inx += 1

            # New PCD
            if azimuth_prev-azimuth > 0:
                if pcd_length < min_pcd_len:
                    azimuth_prev = azimuth
                    distance_packages = np.full(distance_shape, np.nan)
                    pcd_length = 0
                    continue

                # Process PCD...

                # Average the two firing sequence
                distance_packages = np.reshape(distance_packages, (package_num,2,16,4))
                mask = ~np.isnan(distance_packages)
                distance_packages = np.mean(distance_packages, axis=1, where=mask)

                # Average package
                distance_packages = np.mean(distance_packages, axis=-1, where=(mask[:,0] | mask[:,1]))

                distance_list.append(distance_packages)

                # Average the value of the 16 rays
                pcd = np.average(distance_packages, axis=1, weights=weights[order])

                if len(differences):
                    differences.append(pcd_prev-pcd)
                else:
                    differences.append(np.zeros(package_num))

                m=np.mean(differences, axis=0, where=~np.isnan(differences))

                # Velocity
                vel = m/delta_t
                
                # 1/time to collision: velocity/distance
                ttc = vel/pcd
                ttc_list.append(ttc)
                
                # ...Process PCD end

                pcd_prev = pcd
                distance_packages = np.full(distance_shape, np.nan)
                                
                pcd_length = 0
                inx = 0
            # endif
                    
            pcd_length += 1
            
            azimuth_prev = azimuth
            prev_package = current_package
        # endfor
        
        return np.asarray(distance_list), np.asarray(ttc_list)
    

    def heatmap_colors(self, package_num):
        if self.is_empty() or package_num<1:
            return
        
        _, ext = os.path.splitext(self.file_path)

        if ext != ".pcap":
            return
        
        prc = ProcessUnit(package_num)

        distance_list, ttc_list = self.__calc_ttc(package_num, prc)
        thetas = np.deg2rad([-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15])
        azimtuhs = np.linspace(0, 2*np.pi, package_num, endpoint=False)

        coordinate_list = prc.calc_spherical_coords(distance_list, thetas, azimtuhs)

        # ttc
        ttc = np.repeat(ttc_list, 16, axis=1)

        '''
        # HEATMAP METHOD:
        np.save('ttc.npy',ttc)
        max_val = 0.05
        min_val = np.nanmin(ttc)
        norm_ttc = np.zeros_like(ttc)
        norm_ttc = (ttc-min_val)/(max_val-min_val)
        norm_ttc[norm_ttc > 1] = 1
        #hsv_color[:,:,0] = norm_ttc
        #rgb_color = matplotlib.colors.hsv_to_rgb(hsv_color)'''

        rgb_color = np.ones((len(ttc), package_num*16, 3))
        rgb_color[:] = (0.482, 0.678, 0.988)
        rgb_color[ttc > 0.05] = (0.769, 0.071, 0)
        
        # Calculate XYZ coordinates
        for points in coordinate_list:
            points[:, 0], points[:, 1], points[:, 2] = points[:, 0]* np.cos(points[:, 2]) * np.sin(points[:, 1]),\
                                                np.abs(points[:, 0]) *  (np.cos(points[:, 2])) *  np.cos(points[:, 1]),\
                                                points[:, 0]* np.sin(points[:, 2])
            
        
        print("Ready")
        # Set attributes
        self.cloud_arrays_heatmap = coordinate_list
        self.color_arrays = rgb_color
        self.line_line_list = []
        self.line_point_list = []
        
