import argparse
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import choices
from utils import crossover, mutation, find_size_by_name

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--Template", help="the template path")

args = argParser.parse_args()

### using more resonable genetic algorithms to generate pieces

xml_files = glob('./ori_population/*.xml')
xml_list = []
for xml_file in tqdm(xml_files):
    tree = ET.parse(xml_file)
    xml_list.append(tree)

print(len(xml_files))

fitness_list = []
for tree in tqdm(xml_list):
    tree.write('./temp.xml')
    output = os.popen(f'./bin/burrTxt -d ./temp.xml').read()
    output = output.split('-------------------------------------------------------')
    levels = output[-1].split('\n')[1:-2]
    levels = [x.replace('level: ','') for x in levels]
    levels = [sum([int(y) for y in x.split('.')]) for x in levels]
    fitness = np.mean(levels)
    fitness_list.append(fitness)

fitness_list = np.array(fitness_list)
iteration = 0
while True:
    iteration += 1
    sort_indexs = np.argsort(fitness_list)[::-1]
    with open('./logs.txt' ,'a') as f:
        f.write(f'current max levels: {max(fitness_list)}\n')

    ### select top 10 xml as elite
    elite_xmls = []
    for index in sort_indexs[:10]:
        elite_xmls.append(xml_list[index])
    ### save the first elite
    elite_xmls[0].write(f'./results/{iteration}.xml')
    ### keep half of the xmls as parents

    prob = fitness_list/np.sum(fitness_list)

    offsrping_number = 990
    offspring_size = 0
    offspring_list = []
    while True:
        parent_a = choices(xml_list, weights=prob, k=1)
        parent_b = choices(xml_list, weights=prob, k=1)
        xml = crossover(parent_a[0], parent_b[0], args.Template)
        xml.write('./temp.xml')
        output = os.popen('./bin/burrTxt -d -q ./temp.xml').read()
        output = output.split(' ')
        print(output)
        if output[3] != '0':
            offspring_list.append(xml)
            offspring_size += 1
            if offspring_size > offsrping_number:
                break
            print(offspring_size / offsrping_number * 100)

    for i in tqdm(range(len(offspring_list))):
        if np.random.uniform(0,1) < 0.2:
            # 20% mutation
            while True:
                xml = mutation(offspring_list[i], args.Template)
                xml.write('./temp.xml')
                output = os.popen('./bin/burrTxt -d -q ./temp.xml').read()
                output = output.split(' ')
                if output[3] != '0' and output[3] != 'be':
                    print(output)
                    offspring_list[i] = xml
                    break


    fitness_list = []
    xml_list = elite_xmls + offspring_list
    for tree in tqdm(xml_list):
        tree.write('./temp.xml')
        output = os.popen(f'./bin/burrTxt -d ./temp.xml').read()
        output = output.split('-------------------------------------------------------')
        levels = output[-1].split('\n')[1:-2]
        levels = [x.replace('level: ','') for x in levels]
        levels = [sum([int(y) for y in x.split('.')]) for x in levels]
        # fitness = max(levels)
        fitness = np.mean(levels)
        fitness_list.append(fitness)
    with open('./logs.txt' ,'a') as f:
        f.write(f'current max levels: {max(fitness_list)}\n')