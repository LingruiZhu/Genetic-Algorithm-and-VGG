# VGG network currently works well under the keras 2.1.0

import math
import utils
import numpy as np
import pandas as pd

from Initialization import initial_genotype
from evaluation import evaluate
from selection import select
from recombination import recombine, new_register
from Mutuation import mutate
from tools import dict_print, key_print, save_geno
from Genetic_algo import genetic_algo

# check git if works well on pc44


# parameters
# pop_size = 10
# layer_length = 7
# survive_size = math.ceil(pop_size/2.0)
# prob_activate = 0.6
# num_iteration = 50


# if __name__ == "__main__":
#
#     # initialize
#     log_path = 'C:/Users/Admin/PycharmProjects/genetic-algorithm/GA_log.csv'
#     geno = initial_genotype(pop_size, layer_length, [128, 1024], prob_activate)
#     print("===== Initialization part =====")
#     acc = np.zeros((num_iteration, pop_size))
#     df_idx = 0
#
#     geno_log = pd.DataFrame(columns=('iteration', 'geno_type', 'score', 'L0', 'L1', 'L2', 'L3', 'L4'))
#     # dict_print(geno)
#
#     for i in range(num_iteration):
#         print("===================================== %d th iteration =============================" %i)
#         # iterate 5 steps
#         # evaluate and select
#         geno = evaluation(geno, i, num_train=4000, num_test=1000)
#         geno_log = save_geno(geno_log, geno, i)
#         key_print(geno)
#         geno_selected = select(geno, survive_size)
#         print("===== Selection part =====")
#         print('reserved genos: ')
#         key_print(geno_selected)
#
#         # recombine
#         print("====== Recombination Part ======-")
#         new_birth_size = pop_size - int(survive_size)
#
#         for j in range(new_birth_size):
#             parents_list = select(geno_selected, 2, parents_flag=True)
#             parent1 = parents_list[0]
#             parent2 = parents_list[1]
#             child = recombine(parent1, parent2, 0.5, single_child_flag=True)
#             new_register(geno_selected, child)
#
#
#         for key in geno_selected:
#             geno_selected[key] = mutate(geno_selected[key])
#         print("===== Mutation part =====")
#         dict_print(geno_selected)
#         geno = geno_selected
#
#         # convert geno into network structure
#         # input('press enter to build the dense network')
#         # for n in range(num_networks):
#         #     geno_key = random.choice(list(geno.keys()))
#         #     print('Build network according to ' + str(geno_key))
#         #     dict_print(geno[geno_key])
#         #     model = geno2model(geno[geno_key])
#         #
#         #     model.summary()
#
#     # print(geno_log)
#     geno_log.to_csv(log_path, index=False, sep=',')

if __name__ == '__main__':
    pop_size = [8, 12, 16]
    layer_length = [5, 7, 9]
    num_iteration = [30, 50, 70]
    prob_activate = [0.6, 0.7, 0.8]
    prob_mutation = [0.2, 0.3, 0.4]

    for i in range(1):
        genetic_algo(pop_size=4, layer_length=5, num_iteration=10, prob_activate=0.2, prob_mutation=0.4, ex_idx=i)

