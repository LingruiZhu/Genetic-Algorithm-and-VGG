import utils
import numpy as np
import pandas as pd
import math

from Initialization import initial_genotype
from evaluation import evaluate, advanced_evaluate
from selection import select
from recombination import recombine, new_register
from Mutuation import mutate, protect_best
from tools import dict_print, key_print, save_geno


def genetic_algo(pop_size, layer_length, num_iteration, prob_activate, prob_mutation, ex_idx):
    survive_size = math.ceil(pop_size / 2.0)
    file_path = 'C:/Users/Zhu/PycharmProjects/experiment_log/GA_log_0514'
    geno = initial_genotype(pop_size, layer_length, [128, 256], prob_activate)
    geno_log = pd.DataFrame(columns=('iteration', 'geno_type', 'score', 'L0', 'L1', 'L2', 'L3', 'L4'))
    for i in range(num_iteration):
        print("===================================== %d th iteration =============================" %i)
        # iterate 5 steps
        # evaluate and select
        geno = evaluate(geno, i, num_train=6000, num_test=1500, file_path=file_path,save_model=True)

        # if i%10 == 0:
        #     geno = advanced_evaluate(geno)
        geno_log = save_geno(geno_log, geno, i)
        key_print(geno)
        geno_selected = select(geno, survive_size, model='absolute')
        print("===== Selection part =====")
        print('reserved genos: ')
        key_print(geno_selected)

        # recombine
        print("====== Recombination Part ======-")
        new_birth_size = pop_size - int(survive_size)

        for j in range(new_birth_size):
            parents_list = select(geno_selected, 2, model='parents_abs')
            parent1 = parents_list[0]
            parent2 = parents_list[1]
            child = recombine(parent1, parent2, 0.5, single_child_flag=True)
            new_register(geno_selected, child)

        geno_selected = protect_best(geno_selected)
        for key in geno_selected:
            if not geno_selected[key]['protected']:
                if np.random.random() > 0.5:
                    geno_selected[key] = mutate(geno_selected[key], prob_mutation)
        print("===== Mutation part =====")
        dict_print(geno_selected)
        geno = geno_selected


