import numpy as np
import utils


def bool_value(p):
    if np.random.random() < p:
        return True
    else:
        return False


def initial_genotype(pop_size, layer_length, neurons, prob_activate):
    genotype = {}
    global candidate_index
    for i in range(pop_size):
        candidate = {}
        layer_dict = dict()
        active_flag = False
        for j in range(layer_length):
            layer_temp = dict()
            layer_temp['activation'] = bool_value(prob_activate)
            if layer_temp['activation']:
                active_flag = True
            layer_temp['neurons'] = np.random.randint(neurons[0], neurons[1])

            # activation function mapping relationship is like{0: relu, 1: sigmoid, 2: tanhn}
            layer_temp['activation_func'] = np.random.randint(0,2)
            layer_temp['use_bias'] = False
            layer_temp['drop_out'] = False
            layer_temp['drop_out_prob'] = 0.1


            layer_name = 'L' + str(j)
            layer_dict[layer_name] = layer_temp

        if not active_flag:
            layer_dict['L0']['activation'] = True

        candidate['layers'] = layer_dict
        candidate['advanced_score'] = 0
        candidate['score'] = 0.01
        candidate['ad_score'] = 0
        candidate['probability'] = -1
        candidate['flag_change'] = True
        candidate['flag_change_ad'] = True
        candidate['protected'] = False
        candi_name = 'G' + str(i)
        utils.CANDIDATE_IDX = i
        candidate['name'] = candi_name
        genotype[candi_name] = candidate
    return genotype

