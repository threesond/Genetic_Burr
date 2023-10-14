import argparse
import xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage
from utils import array_to_string, find_valid_shape_by_string, generate_base_shape, shrink_one_piece, string_to_array, validate_shape, validate_shape_by_string
import os
import re
from random import choice

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--Template", help="the template path")
argParser.add_argument("-c", "--TotalPieces", type=int, help="total count of distinct pieces to be generated")

args = argParser.parse_args()

input_puzzle_template_file = args.Template
tree = ET.parse(input_puzzle_template_file)
root = tree.getroot()

pieces_dict = dict()

for voxel in root.iter('voxel'):
    voxel_dict = voxel.attrib
    if voxel_dict['name'] != 'Goal':
        if voxel_dict['name'] not in pieces_dict:
            pieces_dict[voxel_dict['name']] = set()
        oz = int(voxel_dict['z'])
        oy = int(voxel_dict['y'])
        ox = int(voxel_dict['x'])
        template_string = voxel.text
        while True:
            template_shape = string_to_array(template_string, (oz,oy,ox))
            ori_shape = template_shape.shape
            template_shape = template_shape.flatten()
            target_shape = np.ones_like(template_shape)

            count = 0
            while True:
                template_shape[target_shape!=1] = 0
                pos = np.argwhere(template_shape == 2)
                pos = np.array(pos)
                pos.shape = (-1)
                sampled_pos = choice(list(pos))
                target_shape[sampled_pos] = 0
                target_shape.shape = ori_shape
                validation = validate_shape(target_shape)
                target_shape = target_shape.flatten()
                if not validation:
                    target_shape[sampled_pos] = 1
                    count += 1
                if count > 100:
                    result_string = array_to_string(target_shape)
                    pieces_dict[voxel_dict['name']].add(result_string)
                    print(result_string)
                    break
            if len(pieces_dict[voxel_dict['name']])>args.TotalPieces:
                break


# basename = os.path.basename(args.Template)
# basename = basename.replace('.xml','')
# save_path = os.path.join('./seed', f'{basename}.npz')
# for key in pieces_dict:
#     pieces_dict[key] = list(pieces_dict[key])
# np.savez(save_path, **pieces_dict)
