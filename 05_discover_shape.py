import argparse
from glob import glob
import os
import numpy as np
import xml.etree.ElementTree as ET
from utils import find_crossover, find_fitness, find_mutation, find_population_by_mutation, find_unique_xmls, rotate_islands, up_mutation
from torch.utils.tensorboard import SummaryWriter

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--Template", help="the template path")
args = argParser.parse_args()

command = 'rm -rf ./exp'
os.system(command)
tb = SummaryWriter(os.path.join('exp', 'tensorboard'))
command = 'rm ./results/*'
os.system(command)

xml = ET.parse(args.Template)
idx = 0
while True:
    xml = up_mutation(xml, args.Template)
    xml.write('/mmfs/temp.xml')
    output = os.popen('./bin/burrTxt -d -q /mmfs/temp.xml').read()
    output = output.split(' ')
    if output[3] != '0' and output[3] != 'be' and output[3] != 'many' and output[3] != 'few':
        print(output)
        xml.write(f'./results/{idx}.xml')
        idx += 1