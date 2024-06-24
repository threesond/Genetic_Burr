import argparse
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import choice, choices
from utils import array_to_string, crossover, mutation, find_size_by_name, string_to_array, validate_shape_by_string

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--Template", help="the template path")
args = argParser.parse_args()

tree = ET.parse(args.Template)
root = tree.getroot()
while True:
    for voxel in root.iter('voxel'):
        voxel_dict = voxel.attrib
        if voxel_dict['name'] != 'Goal':
            oz = int(voxel_dict['z'])
            oy = int(voxel_dict['y'])
            ox = int(voxel_dict['x'])
            voxel_array = string_to_array(voxel.text, (oz, oy, ox))
            while True:
                t_voxel_array = voxel_array.copy()
                pos = np.argwhere(t_voxel_array == 1)
                pos = [tuple(x) for x in pos]
                t_pos = choice(pos)
                z, y, x = t_pos
                t_voxel_array[z, y, x] = 0
                t_string = array_to_string(t_voxel_array)
                if validate_shape_by_string(''.join(t_string), (oz, oy, ox)):
                    break
            voxel.text = ''.join(t_string)
            tree.write('/mmfs/temp.xml')
            output = os.popen('./bin/burrTxt -d -q /mmfs/temp.xml').read()
            output = output.split(' ')
            if output[3] != '0' and output[3] != 'be' and output[3] != 'many' and output[3] != 'few':
                save_name = os.path.join('./aug_results', 'result.xml')
                tree.write(save_name)
                break
            # else:
            #     voxel.text = array_to_string(voxel_array)