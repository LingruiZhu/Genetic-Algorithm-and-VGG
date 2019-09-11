import csv
from tools import dict_print

def csv2geno(csv_file):
    geno = dict()
    csvFile = open(csv_file)
    reader = csv.DictReader(csvFile)
    for row in reader:
        candidate = line2geno(row, 5)
        geno[candidate['name']] = candidate

    return geno


def line2geno(row_dict, num_layers):
    candidate = dict()
    # define layers
    layers = dict()
    for i in range(num_layers):
        layer_temp = dict()
        layer_name = 'L' + str(i)
        layer_temp['activation'] = str2bool(row_dict[layer_name+'-activation'])
        layer_temp['neurons'] = int(row_dict[layer_name+'-neurons'])
        layer_temp['activation_func'] = int(row_dict[layer_name+'-function'])
        layer_temp['use_bias'] = str2bool(row_dict[layer_name+'-bias'])
        layer_temp['drop_out'] = str2bool(row_dict[layer_name+'-drop_out'])
        layer_temp['drop_out_prob'] = float(row_dict[layer_name+'-drop_out_rate'])
        layers[layer_name] = layer_temp

    candidate['layers'] = layers
    candidate['advanced_score'] = 0
    candidate['score'] = row_dict['score']
    candidate['ad_score'] = 0
    candidate['probability'] = -1
    candidate['flag_change'] = True
    candidate['flag_change_ad'] = True
    candidate['protected'] = False
    candi_name = 'G' + str(i)
    candidate['name'] = row_dict['geno_type']

    return candidate

def str2bool(str):
    return True if str.lower() == 'true' else False

# csv_path = 'C:/Users/Zhu/PycharmProjects/experiment_log/GA_log_0430.csv'
# geno = csv2geno(csv_path)
# dict_print(geno)