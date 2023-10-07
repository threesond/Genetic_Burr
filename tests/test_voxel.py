import xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage
from utils import validate_shape

puzzle_file = './examples/puzzle'

tree = ET.parse(puzzle_file)
root = tree.getroot()

# for voxel in root.iter('voxel'):
#     print(voxel.text)

voxel_dict = {
    '#': 1,
    '_': 0
}

voxel_string = '#_____##_____##_###_###____##____###_____##_____##_____#'
voxel_int = [voxel_dict[x] for x in voxel_string]
voxel_array = np.array(voxel_int)

# voxel = np.array(list('#_____##_____#########_____##_____##_____##__#__##_____#'))

voxel_array.shape = (2,4,7)
print(voxel_array)

validate_shape(voxel_array)

# print('-------------')

# kernel = np.zeros([3,3,3], dtype=int)
# kernel[0] = np.array([[0,0,0],[0,1,0],[0,0,0]])
# kernel[1] = np.array([[0,1,0],[1,0,1],[0,1,0]])
# kernel[2] = np.array([[0,0,0],[0,1,0],[0,0,0]])


# result = ndimage.convolve(voxel_array, kernel, mode='constant', cval=0.)
# result[result>1] = 1
# print(result)
# print('--------')
# result = ndimage.convolve(result, kernel, mode='constant', cval=0.)
# result[result>1] = 1
# print(result)


# print(voxel_array)

# sum_result = np.sum(voxel_array, axis=(1,2))
# print(sum_result)