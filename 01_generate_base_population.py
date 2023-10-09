import argparse
import xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage
from utils import array_to_string, find_valid_shape_by_string, generate_base_shape, string_to_array, validate_shape
import os
import re

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--Template", help="the template path")

args = argParser.parse_args()

### first we generate the base population for later use in the genetic algorithm
### all the piece should be valid, that is, they should be all connected shape

# set minimum needed voxel size for each shape
minimum_size_dict = {
    'Frame': 0,
    'A': 9,
    'B': 9
}
additional_size = 9
frame_additional_size = 9
population_size = 1000

validated_size = 0



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
            minimum_size = minimum_size_dict[name]
            if name == 'Frame':
                required_size = np.random.randint(0, frame_additional_size)
            else:
                required_size = np.random.randint(0, additional_size)
            new_shape = find_valid_shape_by_string(
                voxel_string,
                (z,y,x),
                required_size
            )
            new_voxel_string = array_to_string(new_shape)
            voxel.text = new_voxel_string

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