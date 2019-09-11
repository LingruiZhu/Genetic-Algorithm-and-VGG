import numpy as np


def mutate(candidate, p_mut=0.3, neurons_range=[-64, 64], minimum_neurons=64, maximum_neurons=512):
    """
    :param candidate: candidate to mutate
    :param p1: probability to change activation state
    :param p2: probability to change number of neurons
    :param neurons_range: changing range of number of neurons
    :return: mutated candidate
    """

    layers = candidate['layers']
    flag_act = False
    for i in range(len(layers)):
        layer_name = 'L' + str(i)
        layer = layers[layer_name]

        if np.random.random() < p_mut:
            layer['activation'] = not layer['activation']

        neurons_change = np.random.randint(neurons_range[0], neurons_range[1])
        if np.random.random() < p_mut:
            candidate['flag_change'] = True
            layer['neurons'] = layer['neurons'] + neurons_change
            # just to avoid number of neurons becomes zero
            if layer['neurons'] < minimum_neurons:
                layer['neurons'] = minimum_neurons
            if layer['neurons'] > maximum_neurons:
                layer['neurons'] = maximum_neurons
        if np.random.random() < p_mut:
            candidate['flag_change'] = True
            layer['use_bias'] = not layer['use_bias']
        if np.random.random() < p_mut:
            candidate['flag_change'] = True
            layer['drop_out'] = not layer['drop_out']
        if np.random.random() < p_mut:
            candidate['flag_change'] = True
            layer['drop_out_prob'] = np.random.uniform(0.1, 0.3)
        if np.random.random() < p_mut:
            candidate['flag_change'] = True
            layer['activation_func'] = np.random.randint(0,2)

        layers[layer_name] = layer
        # check whether all layers are deactivated, if so set at least one layer activated
        if layer['activation']:
            flag_act = True
        if i == len(layers)-1 & (not flag_act):
            layer['activation'] = True
    candidate['layers'] = layers

    return candidate


def protect_best(genotype):
    best_score = 0
    best_candidate = ''
    for key in genotype:
        genotype[key]['protected'] = False
        if genotype[key]['score'] > best_score:
            best_score = genotype[key]['score']
            best_candidate = key

    if best_candidate != '':
        genotype[key]['protected'] = True
        genotype[key]['flag_change'] = False
    print('=========== the best genotype is '+key+'===========================')
    return genotype
