import unittest
from read_lidar import LidarProcesser, State
import numpy as np

class LidarTest(unittest.TestCase):
    def setUp(self):
        self.prc = LidarProcesser()

    def test_for_empty(self):
        self.assertIs(self.prc.save("file_name.pcd", 0), None)
        self.assertTrue(self.prc.is_empty())
        self.assertEqual(self.prc.get_status(), State.EMPTY)

        img = self.prc.render_image([])
        self.assertEqual(len(img), 480)
        mask = img[img != 255]
        self.assertFalse(mask.any())

        self.assertEqual(self.prc.get_color_array(0,1), [])
        self.assertEqual(self.prc.get_array(0,1), [])
        self.assertEqual(self.prc.get_extents(), (0,0,0))
        self.assertEqual(self.prc.get_avg_pcd_len(), 0)
        self.assertEqual(self.prc.get_first_frame(), [])
        self.assertIs(self.prc.heatmap_colors(0), None)

        self.prc.set_status(State.DEFAULT) # if the state is empty it can't be changed until new file is read
        self.assertTrue(self.prc.is_empty())

    def test_failed_read(self):
        self.assertIs(self.prc.new_file(()), None)
        self.assertTrue(self.prc.is_empty())
        self.assertIs(self.prc.new_file(""), None)
        self.assertTrue(self.prc.is_empty())
        
        # invalid extension
        self.assertIs(self.prc.new_file("alma.txt"), None)
        self.assertTrue(self.prc.is_empty())

        # non-existing file
        self.assertIs(self.prc.new_file("alma.pcd"), None)
        self.assertTrue(self.prc.is_empty())

        self.assertIs(self.prc.new_file("alma.bag"), None)
        self.assertTrue(self.prc.is_empty())

        self.assertIs(self.prc.new_file("alma.pcap"), None)
        self.assertTrue(self.prc.is_empty())

        # read bag file with bad topic
        self.assertIs(self.prc.new_file("./resources/lasrscan_type.bag"), None)
        self.assertTrue(self.prc.is_empty())

        
    def test_reading_and_algorithms(self):
        self.prc.new_file("./resources/data.pcap")
        self.assertFalse(self.prc.is_empty())
        self.assertNotEqual(self.prc.get_frame_num(),0)

        self.prc.set_status(State.GROUND_FILTER)
        self.assertEqual(len(self.prc.get_array(0,1)), 0)
        self.prc.filter_ground(0,1,10,0.1)
        self.assertNotEqual(len(self.prc.get_array(0,1)), 0)

        self.assertEqual(self.prc.get_cluster_num(), 1)
        ex = self.prc.get_extents()
        self.prc.kmeans_colors(3, 0,1, ex)
        self.assertEqual(self.prc.get_cluster_num(), 3)

        self.prc.none_colors(0,1)


        self.prc.set_status(State.HEATMAP)
        colors = self.prc.get_color_array(0,1)

        mask0 = ~np.isnan(colors[:,:,0])
        mask1 = colors[:,:,0] != 1
        mask2 = colors[:,:,1] != 0.706
        mask3 = colors[:,:,2] != 0
        
        mask = (mask0 & ((mask1 | mask2) | mask3))
        self.assertFalse(mask.any())

        self.assertIs(self.prc.heatmap_colors(900), None)
        colors = self.prc.get_color_array(0,1)

        mask0 = ~np.isnan(colors[:,:,0])
        mask1 = colors[:,:,0] != 1
        mask2 = colors[:,:,1] != 0.706
        mask3 = colors[:,:,2] != 0
        
        mask = (mask0 & ((mask1 | mask2) | mask3))
        self.assertTrue(mask.any())



    def test_heatmap_nonpcap(self):
        self.prc.new_file("./resources/next_to_in_parallel.bag")
        self.assertFalse(self.prc.is_empty())
        self.assertNotEqual(self.prc.get_frame_num(),0)

        self.prc.set_status(State.HEATMAP)
        colors = self.prc.get_color_array(0,1)

        mask0 = ~np.isnan(colors[:,:,0])
        mask1 = colors[:,:,0] != 1
        mask2 = colors[:,:,1] != 0.706
        mask3 = colors[:,:,2] != 0
        
        mask = (mask0 & ((mask1 | mask2) | mask3))
        self.assertFalse(mask.any())

        self.assertIs(self.prc.heatmap_colors(900), None)
        colors = self.prc.get_color_array(0,1)

        mask0 = ~np.isnan(colors[:,:,0])
        mask1 = colors[:,:,0] != 1
        mask2 = colors[:,:,1] != 0.706
        mask3 = colors[:,:,2] != 0
        
        mask = (mask0 & ((mask1 | mask2) | mask3))
        self.assertFalse(mask.any())

    def test_saving_and_dbscan(self):
        self.prc.new_file("./resources/pointcloud_1.pcd")
        ex = self.prc.get_extents()
        self.prc.dbscan_colors(0.6, 10, 0, 1, ex)
        colors_before = self.prc.get_color_array(0,1)[0][:,:3]
        #print(colors_before)
        self.prc.save('./resources/output.pcd', 0)
        
        self.prc.new_file("./resources/output.pcd")
        colors_after = self.prc.get_color_array(0,1)[0][:,:3]
        self.assertEqual(len(colors_before), len(colors_after))
        self.assertTrue(((colors_before == colors_after) | (np.isnan(colors_before) & np.isnan(colors_after))).all())
        #print(mask.any())



if __name__ == '__main__':
    unittest.main()
