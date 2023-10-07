from glob import glob
from tqdm import tqdm
import os
import numpy as np
import xml.etree.ElementTree as ET

from utils import crossover, mutation

xml_files = glob('./ori_population/*.xml')
xml_list = []
for xml_file in tqdm(xml_files):
    tree = ET.parse(xml_file)
    xml_list.append(tree)

# xml = crossover(xml_list[0], xml_list[1], './examples/puzzle')
# xml.write('./temp.xml')
xml = mutation(xml_list[0])
xml.write('./temp.xml')