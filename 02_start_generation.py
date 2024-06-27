import argparse
from glob import glob
import os
import numpy as np
import xml.etree.ElementTree as ET
from utils import find_crossover, find_fitness, find_mutation, find_population_by_mutation, find_unique_xmls, rotate_islands
from torch.utils.tensorboard import SummaryWriter

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--Template", help="the template path")
argParser.add_argument("-s", "--ShrinkFrame",
                       action='store_true', help="shrink the frame in mutation")
argParser.add_argument("-c", "--TotalPices", type=int,
                       default=100,  help="total piece of population to generate")
argParser.add_argument("-i", "--IslandNumber", type=int,
                       default=10, help="total island number")

args = argParser.parse_args()

command = 'rm -rf ./exp'
os.system(command)
tb = SummaryWriter(os.path.join('exp', 'tensorboard'))
command = 'rm ./results/*'
os.system(command)

ori_xml_files = glob('./ori_population/*.xml')

island_list = []

for i in range(args.IslandNumber):
    # xml_list = sample_tree_from_files(ori_xml_files, k=100)
    xml_list = find_population_by_mutation(
        ET.parse(args.Template), args.Template, args.ShrinkFrame, 100)
    print(f'initial island {i} generated')
    fitness_list = find_fitness(xml_list)
    island_list.append({
        'xml': xml_list,
        'fitness': fitness_list
    })


iteration = 0
while True:
    iteration += 1
    elite_group_list = []
    for i in range(args.IslandNumber):
        island_dict = island_list[i]
        fitness_list = island_dict['fitness']
        xml_list = island_dict['xml']
        sort_indexs = np.argsort(fitness_list)[::-1]
        elite_xmls = []
        for index in sort_indexs[:2]:
            elite_xmls.append(xml_list[index])
        island_dict['elite'] = elite_xmls
        elite_group_list.append(elite_xmls[0])

    elite_fitness_list = find_fitness(elite_group_list)
    elite_xml = elite_group_list[np.argmax(elite_fitness_list)]
    elite_xml.write(f'./results/{iteration}.xml')

    if iteration % 20 == 0:
        island_list = rotate_islands(island_list)

    max_level_list = []
    for i in range(args.IslandNumber):
        island_dict = island_list[i]
        fitness_list = island_dict['fitness']
        xml_list = island_dict['xml']
        elite_xmls = island_dict['elite']
        offspring_list = find_crossover(
            xml_list, fitness_list, args.TotalPices, args.Template)
        offspring_list = find_mutation(
            offspring_list, args.Template, args.ShrinkFrame)
        xml_list = elite_xmls + offspring_list
        xml_list = find_unique_xmls(xml_list)
        fitness_list = find_fitness(xml_list)
        island_dict['xml'] = xml_list
        island_dict['fitness'] = fitness_list
        max_level_list.append(max(fitness_list))
    tb.add_scalar(f"MaxLevels", max(max_level_list), iteration)
    tb.add_scalar(f"MinLevels", min(max_level_list), iteration)

    # elite_xmls[0].write(f'./results/{iteration}.xml')
