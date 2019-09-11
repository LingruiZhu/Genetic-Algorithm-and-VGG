import numpy as np
import utils
from copy import deepcopy


def recombine(parent1, parent2, prob=0.3, prob_single=0.5, single_child_flag=False):
    """
    :param parent1 : one parent
    :param parent2 : another one
    :param prob: probability to cross over
    :param prob_single: probability used when generating single child
    :param single_child_flag: to decide whether to generate one child or not
    :return: child candidates
    """
    # define layers
    # I changed something in gitlab
    # test2 we changed somthing pzcharm locally
    layers1 = deepcopy(parent1['layers'])
    layers2 = deepcopy(parent2['layers'])
    child1 = dict()
    child2 = dict()
    if not single_child_flag:
        for i in range(len(layers1)):
            p = np.random.random()
            layer_name = 'L'+str(i)
            if p > prob:
                layer_temp = layers1[layer_name]
                layers1[layer_name] = layers2[layer_name]
                layers2[layer_name] = layer_temp
        child1['layers'] = layers1
        child2['layers'] = layers2
        child1 = parameter_initialize(child1)
        child2 = parameter_initialize(child2)
        return child1, child2
    else:
        only_child = dict()
        layers = dict()
        for i in range(len(layers1)):
            layer_name = 'L'+str(i)
            if np.random.random() > prob_single:
                layer_temp = layers1[layer_name]
            else:
                layer_temp = layers2[layer_name]
            layers[layer_name] = layer_temp
        only_child['layers'] = layers
        only_child = parameter_initialize(only_child)
        return only_child


def parameter_initialize(candidate):
    candidate['prob_range'] = [0, 0]
    candidate['name'] = None
    candidate['score'] = 0
    candidate['ad_score'] = 0
    candidate['flag_change'] = True
    candidate['probability'] = 0
    return candidate


def new_register(geno, child):
    utils.CANDIDATE_IDX += 1
    name = 'G' + str(utils.CANDIDATE_IDX)
    child['name'] = name
    geno[name] = child
