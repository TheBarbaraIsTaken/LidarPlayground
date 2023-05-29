import numpy as np
from binascii import hexlify
from os.path import splitext

class PcapReader:
    def __init__(self, file_path):
        _, f_extension = splitext(file_path)
        if (f_extension != ".pcap"):
            raise ValueError("Invalid file extension!\nGot: " + f_extension + ", expected: .pcap.")

        self.file_path = file_path

    def __str__(self) -> str:
        return "Pcap file path: " + self.file_path

    def read_file(self):
        # read the binary file into np array
        data = np.fromfile(self.file_path, dtype=np.uint8)

        # convert the np array to a bytes object
        data_bytes = data.tobytes()

        # convert the bytes object to a hexadecimal string
        self._hex_data = hexlify(data_bytes).decode('utf-8')

    @property
    def raw_data(self):
        return self._hex_data
    
    @property
    def blocks(self):
        return self._hex_data.split('ffee')[1:]


class ProcessUnit:
    def __init__(self, package_num, order=None):
        '''
            order: list of integers, the firing order of lasers
        '''
        if order is None:
            self.order = np.asarray([15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0])
        else:
            self.order = order
        
        self.package_num = package_num


    def calc_azimuth(self, block):
        return int(block[2:4] + block[:2], 16)/100
    
    def is_valid_azimuth(self, azimuth, previous_azimuth):
        return (azimuth <= 360) and not (previous_azimuth - azimuth > 0 and (previous_azimuth < 359 or azimuth > 1))
    
    def calc_package_index(self, azimuth):
        return int(azimuth/(360/self.package_num))
    
    def calc_block_values(self, block):
        return np.asarray([int(block[i+2:i+4]+block[i:i+2], 16) for i in range(4,len(block)-(len(block)-4)%6,6)])[:32]*0.002
    
    def calc_spherical_coords(self, distance_list, thetas, azimuths):
        coordinate_list = np.zeros((len(distance_list),16*self.package_num,3))

        # r
        coordinate_list[:,:,0] = distance_list.reshape(len(distance_list),-1)

        # azimuth (phi)
        azimuths = np.repeat(azimuths,16,axis=0)
        coordinate_list[:,:,1] = azimuths

        # theta
        theta_matrix = np.ones((self.package_num,16))
        theta_matrix[:] = thetas
        theta_matrix = theta_matrix.flatten()
        coordinate_list[:,:,2] = theta_matrix

        return coordinate_list
    
