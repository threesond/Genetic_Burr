import numpy as np
from random import sample
from unionfind import unionfind

voxel_dict = {
    '#': 1,
    '_': 0,
    '+': 2
}

array_dict = {
    1: '#',
    0: '_',
    2: '+'
}

kernel = np.zeros([3,3,3], dtype=int)
kernel[0] = np.array([[0,0,0],[0,1,0],[0,0,0]])
kernel[1] = np.array([[0,1,0],[1,0,1],[0,1,0]])
kernel[2] = np.array([[0,0,0],[0,1,0],[0,0,0]])

def string_to_array(input_shape_string, input_shape_sizes):
    """A simple function to convert burr-tools xml voxel shape string to numpy array

    Args:
        input_shape_string (str): input voxel string in xml
        input_shape_sizes (tuple): input voxel sizes in xml, should in z,y,x order
    """
    voxel_int = [voxel_dict[x] for x in input_shape_string]
    voxel_array = np.array(voxel_int)
    voxel_array.shape = input_shape_sizes
    return voxel_array

def array_to_string(input_shape_array):
    """the reverse of string to array, use the array to generate xml string

    Args:
        input_shape_array (numpy array): the input shape array
    """
    input_shape_array.shape = (-1)
    input_shape_list = list(input_shape_array)
    input_shape_list = [array_dict[x] for x in input_shape_list]
    return ''.join(input_shape_list)


def validate_shape(input_voxel_array):
    """validate voxel shape, make sure the voxels in the shape is all connected by utilizing union find algorithm

    Args:
        input_voxel_array (numpy array): the input shape in numpy array
    """
    
    pos = np.argwhere(input_voxel_array == 1)
    pos = [tuple(x) for x in pos]
    pos_dict = {item:ind for ind, item in enumerate(pos)}
    u = unionfind(len(pos))
    for t_pos in pos:
        z,y,x = t_pos
        if (z,y,x+1) in pos_dict:
            u.unite(pos_dict[t_pos], pos_dict[(z,y,x+1)])
        if (z,y,x-1) in pos_dict:
            u.unite(pos_dict[t_pos], pos_dict[(z,y,x-1)])
        if (z,y-1,x) in pos_dict:
            u.unite(pos_dict[t_pos], pos_dict[(z,y-1,x)])
        if (z,y+1,x) in pos_dict:
            u.unite(pos_dict[t_pos], pos_dict[(z,y+1,x)])
        if (z-1,y,x) in pos_dict:
            u.unite(pos_dict[t_pos], pos_dict[(z-1,y,x)])
        if (z+1,y,x) in pos_dict:
            u.unite(pos_dict[t_pos], pos_dict[(z+1,y,x)])
    if len(u.groups()) > 1:
        return False
    elif len(u.groups()) == 1:
        return True
        



    
def generate_base_shape(template_shape, voxel_num):
    """generate a base shape by random sampling the viable voxels

    Args:
        template_shape (numpy array): the template shape which have viable voxels
        voxel_num (int): the mininum number for the base shape
    """
    ori_shape = template_shape.shape
    template_shape = template_shape.flatten()
    pos = np.argwhere(template_shape == 2)
    pos = np.array(pos)
    pos.shape = (-1)
    sampled_pos = sample(list(pos), k=voxel_num)
    template_shape[template_shape==2] = 0
    template_shape[sampled_pos] = 1
    template_shape.shape = ori_shape
    return template_shape