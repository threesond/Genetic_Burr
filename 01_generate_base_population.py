import argparse
import xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage
from utils import array_to_string, find_valid_shape_by_string, generate_base_shape, string_to_array, validate_shape
import os
import re
from random import sample

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--Template", help="the template path")
argParser.add_argument("-c", "--PopulationSize", type=int, help="the population size")

args = argParser.parse_args()
population_size = args.PopulationSize
validated_size = 0

basename = os.path.basename(args.Template)
basename = basename.replace('.xml','')
npz_path = os.path.join('./seed', f'{basename}.npz')
pieces_dict = np.load(npz_path, allow_pickle=True)
pieces_dict = dict(pieces_dict)
for key in pieces_dict:
    pieces_dict[key] = list(pieces_dict[key])

while True:
    input_puzzle_template_file = args.Template
    tree = ET.parse(input_puzzle_template_file)
    root = tree.getroot()
    for ind, voxel in enumerate(root.iter('voxel')):
        voxel_dict = voxel.attrib
        z = int(voxel_dict['z'])
        y = int(voxel_dict['y'])
        x = int(voxel_dict['x'])
        voxel_string = voxel.text
        name = voxel_dict['name']
        if name != 'Goal':
            sample_string = sample(pieces_dict[name], k=1)
            new_shape_string = sample_string[0]
            voxel.text = new_shape_string

    tree.write('./temp.xml')

    output = os.popen('./bin/burrTxt -d -q ./temp.xml').read()
    output = output.split(' ')
    if output[3] != '0' and output[3] != 'be' and output[3] != 'few':
        save_name = os.path.join('./ori_population', f'{validated_size}.xml')
        tree.write(save_name)
        validated_size += 1
        print(validated_size / population_size * 100)
        if validated_size > population_size:
            break
        print(output)