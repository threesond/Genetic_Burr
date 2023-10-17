import argparse
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import choice, choices
from utils import array_to_string, crossover, mutation, find_size_by_name, string_to_array, validate_shape_by_string

argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--Piece", help="the piece path")
argParser.add_argument("-s", "--Steps", type=int, help="the minimum steps")

args = argParser.parse_args()


tree = ET.parse(args.Piece)
root = tree.getroot()
while True:
    for voxel in root.iter('voxel'):
        voxel_dict = voxel.attrib
        if voxel_dict['name'] != 'Goal':
            oz = int(voxel_dict['z'])
            oy = int(voxel_dict['y'])
            ox = int(voxel_dict['x'])
            voxel_array = string_to_array(voxel.text, (oz, oy, ox))
            count = 0
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
                else:
                    count += 1
                if count > 100:
                    t_string = array_to_string(voxel_array)
                    break
            voxel.text = ''.join(t_string)
            tree.write('./temp.xml')
            output = os.popen(f'./bin/burrTxt -d ./temp.xml').read()
            output = output.split(
                '-------------------------------------------------------')
            levels = output[-1].split('\n')[1:-2]
            levels = [x.replace('level: ', '') for x in levels]
            levels = [sum([int(y) for y in x.split('.')]) for x in levels]
            fitness = np.mean(levels)
            if fitness >= args.Steps:
                print('augmented 1')
                save_name = os.path.join('./aug_results', 'result.xml')
                tree.write(save_name)
            else:
                voxel.text = array_to_string(voxel_array)
