from glob import glob
from tqdm import tqdm
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import choice
from utils import crossover, mutation

xml_files = glob('./ori_population/*.xml')
xml_list = []
for xml_file in tqdm(xml_files):
    tree = ET.parse(xml_file)
    xml_list.append(tree)

print(len(xml_files))

### first generate the fitness
### fitness is the total moves of the solution

fitness_list = []
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
    xml_size = len(xml_list)
    keep_size = int(xml_size/2)
    parent_xmls = []
    for index in sort_indexs[:keep_size]:
        parent_xmls.append(xml_list[index])

    ### now make 900 crossovers
    offsrping_number = 800
    offspring_size = 0
    offspring_list = []
    while True:
        parent_a = choice(parent_xmls)
        parent_b = choice(parent_xmls)
        xml = crossover(parent_a, parent_b, './examples/puzzle')
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

    ### now make 100 mutations
    mutation_number = 200
    mutation_size = 0
    mutation_list = []
    while True:
        condidate = choice(parent_xmls)
        xml = mutation(condidate, './examples/puzzle')
        xml.write('./temp.xml')
        output = os.popen('./bin/burrTxt -d -q ./temp.xml').read()
        output = output.split(' ')
        if output[3] != '0' and output[3] != 'be':
            print(output)
            mutation_list.append(xml)
            mutation_size += 1
            if mutation_size > mutation_number:
                break
            print(mutation_size / mutation_number * 100)

    fitness_list = []
    xml_list = elite_xmls + offspring_list + mutation_list
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
