from collections import Counter
import os
import numpy as np
from random import choices, sample, choice
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
    
def validate_shape(input_voxel_array, return_suitable_shape=False):
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
    if return_suitable_shape:
        union_lengths = [len(x) for x in u.groups()]
        max_ind = np.argmax(union_lengths)
        max_group = u.groups()[max_ind]
        return_shape = np.zeros_like(input_voxel_array)
        for ind in max_group:
            t_pos = pos[ind]
            return_shape[t_pos] = 1
        return return_shape
    else:
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
                                additional_size,
                                return_suitable_shape=False):
    """find a valid shape by randomly sample the input voxel template

    Args:
        input_voxel_string (str): the input shape template
        input_voxel_shape (tuple): the input shape
        additional_size(int): the additional voxel size
    """
    voxel_array = string_to_array(input_voxel_string, input_voxel_shape)
    if return_suitable_shape:
        base_shape_candidate = generate_base_shape(voxel_array, additional_size)
        shape_array = validate_shape(base_shape_candidate, return_suitable_shape)
        return shape_array
    else:
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

def mutation(xml, template_xml, shrink_frame):
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
                # mutation_times = np.random.randint(1,5)
                mutation_times = 1
                for _ in range(mutation_times):
                    t_pos = choice(pos)
                    z,y,x = t_pos
                    if np.random.uniform(0,1)<0.5:
                        if 'Frame' in name:
                            if shrink_frame:
                                t_voxel_array[z,y,x] = 0
                            else:
                                pass
                        else:
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
        
def shrink_one_piece(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    voxel_list = [x for x in root.iter('voxel')]
    counter = 0
    while True:
        voxel = choice(voxel_list)
        voxel_dict = voxel.attrib
        oz = int(voxel_dict['z'])
        oy = int(voxel_dict['y'])
        ox = int(voxel_dict['x'])
        voxel_array = string_to_array(voxel.text, (oz,oy,ox))
        t_voxel_array = voxel_array.copy()
        pos = np.argwhere(t_voxel_array == 1)
        pos = [tuple(x) for x in pos]
        t_pos = choice(pos)
        z,y,x = t_pos
        t_voxel_array[z,y,x] = 0
        t_string = array_to_string(t_voxel_array)
        if validate_shape_by_string(''.join(t_string), (oz,oy,ox)):
            voxel.text = ''.join(t_string)
            tree.write('./temp2.xml')
            return True
        else:
            counter += 1
        if counter > 100:
            return False
        
def find_if_in_list(xml_list, xml):
    xml_string_list = [ET.tostring(x.getroot(), encoding='unicode') for x in xml_list]
    xml_string = ET.tostring(xml.getroot(), encoding='unicode')
    return xml_string in xml_string_list

def find_unique_xmls(xml_list):
    xml_string_list = [ET.tostring(x.getroot(), encoding='unicode') for x in xml_list]
    c = Counter(xml_string_list)
    unique_list = []
    for xml in xml_list:
        xml_string = ET.tostring(xml.getroot(), encoding='unicode')
        if c[xml_string] == 1:
            unique_list.append(xml)
    counted_list = []
    for xml in xml_list:
        xml_string = ET.tostring(xml.getroot(), encoding='unicode')
        if c[xml_string] > 1:
            if xml_string not in counted_list:
                counted_list.append(xml_string)
                unique_list.append(xml)
    return unique_list

def find_fitness(xml_list):
    fitness_list = []
    for tree in xml_list:
        tree.write('/mmfs/temp.xml')
        output = os.popen(f'./bin/burrTxt -d /mmfs/temp.xml').read()
        output = output.split('-------------------------------------------------------')
        levels = output[-1].split('\n')[1:-2]
        levels = [x.replace('level: ','') for x in levels]
        levels = [sum([int(y) for y in x.split('.')]) for x in levels]
        fitness = np.mean(levels)
        fitness_list.append(fitness)
    fitness_list = np.array(fitness_list)
    fitness_list = np.nan_to_num(fitness_list, nan=0)
    return fitness_list

def find_crossover(xml_list, fitness_list, offspring_number, template):
    # for index in sort_indexs[:10]:
    #     fitness_list[index] = fitness_list[index] * 10

    prob = fitness_list/np.sum(fitness_list)
    # temprature = 0.1
    # exp_list = np.exp(fitness_list/temprature)
    # prob = exp_list / np.sum(exp_list)
    offspring_size = 0
    offspring_list = []
    while True:
        parent_a = choices(xml_list, weights=prob, k=1)
        parent_b = choices(xml_list, weights=prob, k=1)
        xml = crossover(parent_a[0], parent_b[0], template)
        xml.write('/mmfs/temp.xml')
        output = os.popen('./bin/burrTxt -d -q /mmfs/temp.xml').read()
        output = output.split(' ')
        if output[3] != '0' and output[3] != 'be' and output[3] != 'many' and output[3] != 'few':
            offspring_list.append(xml)
            offspring_size += 1
            if offspring_size > offspring_number:
                break
    return offspring_list
        
def find_mutation(offspring_list, template, shrink_frame):
    for i in range(len(offspring_list)):
        if np.random.uniform(0,1) < 0.2:
            # 20% mutation
            while True:
                xml = mutation(offspring_list[i], template, shrink_frame)
                xml.write('/mmfs/temp.xml')
                output = os.popen('./bin/burrTxt -d -q /mmfs/temp.xml').read()
                output = output.split(' ')
                if output[3] != '0' and output[3] != 'be' and output[3] != 'many' and output[3] != 'few':
                    offspring_list[i] = xml
                    break
    return offspring_list

def find_population_by_mutation(ori_xml, template, shrink_frame, total_number):
    count = 0
    output_list = []
    while True:
        xml = mutation(ori_xml, template, shrink_frame)
        xml.write('/mmfs/temp.xml')
        output = os.popen('./bin/burrTxt -d -q /mmfs/temp.xml').read()
        output = output.split(' ')
        if output[3] != '0' and output[3] != 'be' and output[3] != 'many' and output[3] != 'few':
            output_list.append(xml)
            count += 1
        print(count)
        if count > total_number:
            break
    return output_list

def sample_tree_from_files(ori_xml_files, k=100):
    # xml_files = sample(ori_xml_files, k=100)
    xml_files = ori_xml_files * 25
    xml_list = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        xml_list.append(tree)
    return xml_list

def rotate_islands(island_list):
    for i in range(len(island_list)):
        island_dict_target = island_list[i]
        island_dict_source = island_list[i-1]
        
        xml_target = island_dict_target['xml']
        fitness_target = list(island_dict_target['fitness'])
        
        xml_source = island_dict_source['xml']
        fitness_source = list(island_dict_source['fitness'])
        
        inds = list(np.random.randint(0,len(fitness_source),5))
        for j in inds:
            xml_target.append(xml_source[j])
            fitness_target.append(fitness_source[j])
        fitness_target = np.array(fitness_target)
        island_dict_target['xml'] = xml_target
        island_dict_target['fitness'] = fitness_target
    return island_list