import numpy as np


def select(geno_type, n_candidate, model='normal'):
    score_sum = 0
    # two modes: parent mode and normal mode
    # parent mode returns a list of two candidates
    # normal mode returns dictionary including n candidates

    # the loop is for calculating score sum
    for key in geno_type:
        candidate = geno_type[key]
        score_sum = score_sum + candidate['score']

    # calculate probability and cumulative probability of candidates
    start_point = 0.0
    for key in geno_type:
        candidate = geno_type[key]
        if model == 'parents_abs':
            prob = 1 / len(geno_type)
        else:
            prob = candidate['score'] / score_sum
        candidate['probability'] = prob
        candidate['prob_range'] = [start_point, start_point+prob]
        start_point = start_point + prob

    selected_geno = {}
    i = 0
    while 1:
        q = np.random.random()
        print("at current, the iteration in possible dead loop", i)
        print("random num in selection loop", q)
        for key in geno_type:
            candidate = geno_type[key]
            p_range = candidate['prob_range']
            print(p_range[0], p_range[1])
            if (q >= p_range[0]) & (q < p_range[1]):
                selected_geno[key] = geno_type[key]
                continue
        if len(selected_geno) >= n_candidate:
            break
        i = i + 1
    if model == 'normal':
        return selected_geno
    # parents model: return a list of two candidates not dict
    elif n_candidate ==2 and model == 'parents':
        parents_list = []
        for key in selected_geno:
            parents_list.append(selected_geno[key])
        return parents_list

    elif model == 'absolute':
        selected_abs_geno = {}
        score_list = []
        for key in geno_type:
            score_list.append(geno_type[key]['score'])
        mid = np.median(np.array(score_list))
        count = 0
        for key in geno_type:
            if geno_type[key]['score'] >= mid:
                selected_abs_geno[key] = geno_type[key]
                count += 1
            if count == n_candidate:
                continue
        return selected_abs_geno

    elif model == 'parents_abs' and n_candidate == 2:
        parent_list_abs = []
        for key in selected_geno:
            parent_list_abs.append(selected_geno[key])
        return parent_list_abs






