import argparse
import xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage
from utils import array_to_string, find_valid_shape_by_string, generate_base_shape, grow_shape, string_to_array, validate_shape
import os
import re
from random import sample

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
    oz = int(voxel_dict['z'])
    oy = int(voxel_dict['y'])
    ox = int(voxel_dict['x'])
    template_string = voxel.text
    name = voxel_dict['name']
    if name != 'Goal':
        if name not in pieces_dict:
            pieces_dict[name] = set()
            # pieces_dict[name].add(template_string)
        while True:
            # sample_string = choice(list(pieces_dict[name]))
            # print('printing sample string', sample_string)
            # print(pieces_dict[name])
            # sample_string = sample(pieces_dict[name], k=1)
            # sample_string = sample_string[0]
            shape_array = string_to_array(template_string, (oz,oy,ox))
            result_string = grow_shape(shape_array)
            if result_string != '':
                pieces_dict[name].add(result_string)
            if len(pieces_dict[name])>args.TotalPieces:
                break
            print(result_string)

basename = os.path.basename(args.Template)
basename = basename.replace('.xml','')
save_path = os.path.join('./shapes', f'{basename}.npz')
for key in pieces_dict:
    pieces_dict[key] = list(pieces_dict[key])
np.savez(save_path, **pieces_dict)