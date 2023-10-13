import numpy as np
from random import sample, choice
from unionfind import unionfind
import xml.etree.ElementTree as ET

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
    shape_array = input_shape_array.copy()
    shape_array.shape = (-1)
    input_shape_list = list(shape_array)
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
    
def validate_shape_by_string(input_voxel_string, input_shape_size):
    """validate the shape by input string

    Args:
        input_voxel_string (str): input voxel string
        input_shape_size (tuple): the input shape size
    """
    input_voxel_array = string_to_array(input_voxel_string, input_shape_size)
    return validate_shape(input_voxel_array)

def find_valid_shape_by_string(input_voxel_string, 
                                input_voxel_shape, 
                                additional_size):
    """find a valid shape by randomly sample the input voxel template

    Args:
        input_voxel_string (str): the input shape template
        input_voxel_shape (tuple): the input shape
        additional_size(int): the additional voxel size
    """
    voxel_array = string_to_array(input_voxel_string, input_voxel_shape)
    while True:
        base_shape_candidate = generate_base_shape(voxel_array, additional_size)
        validation = validate_shape(base_shape_candidate)
        if validation:
            break
    return base_shape_candidate
    
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

def crossover(xml_a, xml_b, template_xml):
    """make a crossover os parent xmls

    Args:
        xml_a (xml tree): parent a
        xml_b (xml tree): parent b
        template_xml (str): template xml file
    """
    root_a = xml_a.getroot()
    root_b = xml_b.getroot()
    template_tree = ET.parse(template_xml)
    template_root = template_tree.getroot()
    for voxel_a, voxel_b, voxel_template in zip(root_a.iter('voxel'), root_b.iter('voxel'), template_root.iter('voxel')):
        voxel_a_dict = voxel_a.attrib
        voxel_a_string = list(voxel_a.text)
        voxel_b_string = list(voxel_b.text)
        z = int(voxel_a_dict['z'])
        y = int(voxel_a_dict['y'])
        x = int(voxel_a_dict['x'])
        name = voxel_a_dict['name']
        if name != 'Goal':
            while True:
                crosspoint = np.random.randint(len(voxel_a_string))
                t_string = voxel_a_string
                t_string[crosspoint:] = voxel_b_string[crosspoint:]
                voxel_template.text = ''.join(t_string)
                if validate_shape_by_string(voxel_template.text, (z,y,x)):
                    break
    return template_tree

def mutation(xml, template_xml):
    """make a mutation to the voxel by adding or removing a voxel randomly

    Args:
        xml (xml tree): the input candidate
    """
    root = xml.getroot()
    template_tree = ET.parse(template_xml)
    template_root = template_tree.getroot()
    for voxel, voxel_template in zip(root.iter('voxel'), template_root.iter('voxel')):
        voxel_string = voxel.text
        voxel_dict = voxel.attrib
        oz = int(voxel_dict['z'])
        oy = int(voxel_dict['y'])
        ox = int(voxel_dict['x'])
        name = voxel_dict['name']
        if name != 'Goal':
            voxel_array = string_to_array(voxel_string, (oz,oy,ox))
            while True:
                pos = np.argwhere(voxel_array == 1)
                t_voxel_array = voxel_array.copy()
                pos = [tuple(x) for x in pos]
                mutation_times = np.random.randint(1,5)
                for _ in range(mutation_times):
                    t_pos = choice(pos)
                    z,y,x = t_pos
                    if np.random.uniform(0,1)<0.5:
                        t_voxel_array[z,y,x] = 0
                    else:
                        mode = np.random.randint(6)
                        if mode == 0:
                            t_voxel_array[z,y,x-1] = 1
                        if mode == 1:
                            t_voxel_array[z,y,np.minimum(x+1,ox-1)] = 1
                        if mode == 2:
                            t_voxel_array[z,y-1,x] = 1
                        if mode == 3:
                            t_voxel_array[z,np.minimum(y+1,oy-1),x] = 1
                        if mode == 4:
                            t_voxel_array[z-1,y,x] = 1
                        if mode == 5:
                            t_voxel_array[np.minimum(z+1,oz-1),y,x] = 1
                t_string = array_to_string(t_voxel_array)
                if validate_shape_by_string(''.join(t_string), (oz,oy,ox)):
                    voxel_template.text = ''.join(t_string)
                    break
    return template_tree

def find_size_by_name(xml, shape_name):
    """find the frame size by name

    Args:
        xml (xml): the input xml tree
    """
    root = xml.getroot()
    for voxel in root.iter('voxel'):
        voxel_dict = voxel.attrib
        name = voxel_dict['name']
        oz = int(voxel_dict['z'])
        oy = int(voxel_dict['y'])
        ox = int(voxel_dict['x'])
        if name == shape_name:
            voxel_string = voxel.text
            voxel_array = string_to_array(voxel_string, (oz, oy, ox))
            return np.sum(voxel_array)