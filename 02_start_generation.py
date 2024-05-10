import argparse
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import choices, sample
from utils import crossover, find_if_in_list, find_unique_xmls, mutation, find_size_by_name
from torch.utils.tensorboard import SummaryWriter

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--Template", help="the template path")
argParser.add_argument("-s", "--ShrinkFrame", action='store_true', help="shrink the frame in mutation")
argParser.add_argument("-c", "--TotalPices", type=int, help="total piece of population to generate")

args = argParser.parse_args()

command = 'rm -rf ./exp'
os.system(command)

tb = SummaryWriter(os.path.join('exp', 'tensorboard'))
command = 'rm ./results/*'
os.system(command)

### using more resonable genetic algorithms to generate pieces

xml_files = glob('./ori_population/*.xml')
xml_list = []
for xml_file in tqdm(xml_files):
    tree = ET.parse(xml_file)
    xml_list.append(tree)

print(len(xml_files))

fitness_list = []
for tree in tqdm(xml_list):
    tree.write('/mmfs/temp.xml')
    output = os.popen(f'./bin/burrTxt -d /mmfs/temp.xml').read()
    output = output.split('-------------------------------------------------------')
    levels = output[-1].split('\n')[1:-2]
    levels = [x.replace('level: ','') for x in levels]
    levels = [sum([int(y) for y in x.split('.')]) for x in levels]
    fitness = np.mean(levels)
    fitness_list.append(fitness)

fitness_list = np.array(fitness_list)
fitness_list = np.nan_to_num(fitness_list, nan=0)
iteration = 0
while True:
    iteration += 1
    sort_indexs = np.argsort(fitness_list)[::-1]
    # with open('./logs.txt' ,'a') as f:
    #     f.write(f'current max levels: {max(fitness_list)}\n')

    ### select top 10 xml as elite
    elite_xmls = []
    for index in sort_indexs[:2]:
        elite_xmls.append(xml_list[index])
    ### save the first elite
    elite_xmls[0].write(f'./results/{iteration}.xml')
    ### keep half of the xmls as parents
    
    # for index in sort_indexs[:10]:
    #     fitness_list[index] = fitness_list[index] * 10

    # prob = fitness_list/np.sum(fitness_list)
    
    # fitness_mean = np.mean(fitness_list)
    # fitness_std = np.std(fitness_list)
    # fitness_list = (fitness_list - fitness_mean) / (fitness_std + 0.00001)
    
    temprature = 1.
    exp_list = np.exp(fitness_list/temprature)
    prob = exp_list / np.sum(exp_list)

    offsrping_number = args.TotalPices
    offspring_size = 0
    offspring_list = []
    while True:
        parent_a = choices(xml_list, weights=prob, k=1)
        parent_b = choices(xml_list, weights=prob, k=1)
        xml = crossover(parent_a[0], parent_b[0], args.Template)
        xml.write('/mmfs/temp.xml')
        output = os.popen('./bin/burrTxt -d -q /mmfs/temp.xml').read()
        output = output.split(' ')
        # print(output)
        # if output[3] != '0':
        if output[3] != '0' and output[3] != 'be' and output[3] != 'many' and output[3] != 'few':
            offspring_list.append(xml)
            offspring_size += 1
            if offspring_size > offsrping_number:
                break
            print(offspring_size / offsrping_number * 100)

    for i in tqdm(range(len(offspring_list))):
        if np.random.uniform(0,1) < 0.2:
            # 20% mutation
            while True:
                xml = mutation(offspring_list[i], args.Template, args.ShrinkFrame)
                xml.write('/mmfs/temp.xml')
                output = os.popen('./bin/burrTxt -d -q /mmfs/temp.xml').read()
                output = output.split(' ')
                if output[3] != '0' and output[3] != 'be' and output[3] != 'many' and output[3] != 'few':
                    print(output)
                    offspring_list[i] = xml
                    break
    # offspring_list = find_unique_xmls(offspring_list)
    fitness_list = []
    xml_list = elite_xmls + offspring_list
    xml_list = find_unique_xmls(xml_list)
    for tree in tqdm(xml_list):
        tree.write('/mmfs/temp.xml')
        output = os.popen(f'./bin/burrTxt -d /mmfs/temp.xml').read()
        output = output.split('-------------------------------------------------------')
        levels = output[-1].split('\n')[1:-2]
        levels = [x.replace('level: ','') for x in levels]
        levels = [sum([int(y) for y in x.split('.')]) for x in levels]
        # fitness = max(levels)
        fitness = np.mean(levels)
        fitness_list.append(fitness)
    fitness_list = np.array(fitness_list)
    fitness_list = np.nan_to_num(fitness_list, nan=0)
    tb.add_scalar("MaxLevels", max(fitness_list), iteration)
    # with open('./logs.txt' ,'a') as f:
    #     f.write(f'current max levels: {max(fitness_list)}\n')