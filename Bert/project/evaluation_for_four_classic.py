# coding=utf-8
import json

def read_json(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_tsv(input_file):
    probabilities_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            proba_str_list = line.split('\t')
            probabilities_list.append([float(x) for x in proba_str_list])
    return probabilities_list

def get_predict_label(probabilities_list):
    predict_relation_list = []
    for x in probabilities_list:
        predict_relation_list.append(x.index(max(x)))
    return predict_relation_list


def get_relation_list_from_structure(data):
    examples = []
    # every triple
    for triple_data in data:
        # every sentence window
        for x in triple_data['txt']:
            label = triple_data['relation']
            examples.append(label)
    return examples

def get_true_label(true_relation_list_in_str, label_list):
    true_relation_list = []
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    # transform relation to according id
    for x in true_relation_list_in_str:
        true_relation_list.append(label_map[x])

    return true_relation_list

def get_predict_relation_str(predict_relation_list, label_list):
    predict_relation_list_str = [label_list[x] for x in predict_relation_list]
    return predict_relation_list_str



def accuracy(predict_relation_list, true_relation_list):
    assert len(predict_relation_list) == len(true_relation_list)
    example_sum = len(true_relation_list)
    example_correct = 0
    for (predict_rel, true_rel) in zip(predict_relation_list, true_relation_list):
        if predict_rel == true_rel:
            example_correct += 1

    return example_correct / example_sum

def kappa_factor(predict_relation_list, true_relation_list, relation_num, label_list):
    po = accuracy(predict_relation_list, true_relation_list)
    # predict_relation_list, true_relation_list are list of numbers, which represent the according relation list
    true_pre_count = {}
    pe_numerator = 0
    for i in range(relation_num):
        true_num = true_relation_list.count(i)
        pre_num = predict_relation_list.count(i)
        pe_numerator += true_num * pre_num
        true_pre_count[label_list[i]] = {'true_num': true_num, 'pre_num': pre_num}
    pe = pe_numerator / pow(len(true_relation_list), 2)

    return (po - pe) / (1 - pe), true_pre_count


if __name__ == '__main__':
    true_path = './DreamOfTheRedChamber/expand-7-1.5-1.5-without_no_relation_2.0/'
    pre_path = './result/DreamOfTheRedChamber/expand-7-1.5-1.5-without_no_relation_2.0/'

    true_relation_file = true_path + 'test.json'
    label_file = pre_path + 'label_list.json'
    predict_relation_file = pre_path + 'test_results_10.tsv'
    predict_relation_str_file = pre_path + 'test_results_10_str.tsv'
    true_pre_count_file = pre_path + 'true_pre_count.tsv'
    # get the label list
    label_list = read_json(label_file)

    # get the true_relation_list
    true_relation_list_in_str = get_relation_list_from_structure(read_json(true_relation_file))
    true_relation_list = get_true_label(true_relation_list_in_str, label_list)

    # get the probabilities for each example
    # and get label id
    probabilities_list = read_tsv(predict_relation_file)
    predict_relation_list = get_predict_label(probabilities_list)

    # evaluate accuracy
    acc = accuracy(predict_relation_list, true_relation_list)

    # evaluate kappa
    # how many type relations
    relation_num = len(probabilities_list[0])
    kappa, true_pre_count = kappa_factor(predict_relation_list, true_relation_list, relation_num, label_list)

    print('Accuracy:  ' + str(acc))
    print('Kappa Factor:  ' + str(kappa))

    # 将预测的关系按顺序输出
    predict_relation_list_str = get_predict_relation_str(predict_relation_list, label_list)
    with open(predict_relation_str_file, 'w', encoding='utf-8') as fin:
        for x in predict_relation_list_str:
            fin.write(x + '\n')

    # 将预测和真实关系的统计写入文件
    with open(true_pre_count_file, 'w', encoding='utf-8') as fin:
        json.dump(true_pre_count, fin, indent=4, ensure_ascii=False)

