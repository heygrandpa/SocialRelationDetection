# -*- encoding: utf-8 -*-
from openpyxl import load_workbook
import json
import copy
import re

def read_data_from_xlsx_not_expand(file):
    workbook = load_workbook(file)
    booksheet = workbook.active # 当前活跃的sheet，默认第一个sheet
    # 行数据
    rows = booksheet.rows

    Dataset = [] # 数据，每个三元组及其信息占一个元素
    id_list = [] # 已经存在于Dataset中的三元组

    sen_max_len = 0 # 最大句子长度

    # 下标从1开始计数
    i = 0
    for row in rows:
        i += 1
        if i == 1: # 跳过第一行
            continue
        line = [col.value for col in row]

        # 一行数据
        now_relation = line[6]
        new_id = line[4] + '-' + now_relation + '-' + line[5] # head-relation-tail
        if len(line[0]) > sen_max_len:
            sen_max_len = len(line[0])
        if new_id not in id_list: # 是一个全新的三元组
            id_list.append(new_id)
            info = {}
            info['id'] = new_id
            info['sub'] = line[4]
            info['obj'] = line[5]
            info['relation'] = now_relation
            info['txt'] = []
            raw_sentence_arr = re.split(r"<.?head>|<.?tail>", line[0])
            raw_sentence = ''.join(raw_sentence_arr)
            info['txt'].append({'sentence': raw_sentence, 'sentence_marked': line[0]})
            Dataset.append(info)
        else: # 此三元组已经出现过，只需要添加新的句子窗口
            index = id_list.index(new_id)
            # 避免句子窗口样本重复
            has_contain = False
            for x in Dataset[index]['txt']:
                if x['sentence_marked'] == line[0]:
                    has_contain = True
                    break
            if not has_contain:
                raw_sentence_arr = re.split(r"<.?head>|<.?tail>", line[0])
                raw_sentence = ''.join(raw_sentence_arr)
                Dataset[index]['txt'].append({'sentence': raw_sentence, 'sentence_marked': line[0]})

    # count
    triple_num = len(id_list)
    data_num = 0
    for x in Dataset:
        data_num += len(x['txt'])

    print('句子窗口最大长度：', str(sen_max_len))
    print('三元组个数：', str(triple_num))
    print('数据条数：', str(data_num))
    return Dataset



def read_data_from_xlsx_expand(file, expand_file):
    # 读取扩展三元组文件
    with open(expand_file, 'r', encoding='utf-8') as fin:
        expand_data = json.load(fin)
    # 未扩展的
    Dataset = read_data_from_xlsx_not_expand(file)
    expand_Dataset = Dataset

    expand_list = [x['id'] for x in expand_data]
    contained_list = [x['id'] for x in Dataset]

    sum_valid_row = 0 # 统计总共有多少条数据

    # 扩展
    for triple_data in Dataset:
        sum_valid_row += len(triple_data['txt'])
        if triple_data['id'] in expand_list:
            new_triple_id = expand_data[expand_list.index(triple_data['id'])]['add']
            if new_triple_id not in contained_list:
                new_info = {}
                new_info['id'] = new_triple_id
                new_info['sub'] = new_info['id'].split('-')[0]
                new_info['relation'] = new_info['id'].split('-')[1]
                new_info['obj'] = new_info['id'].split('-')[2]
                new_info['txt'] = copy.deepcopy(triple_data['txt'])
                for x in new_info['txt']:
                    x['sentence_marked'] = ''  # 无sentence_marked
                contained_list.append(new_triple_id)
                expand_Dataset.append(new_info)
                sum_valid_row += len(triple_data['txt'])

    print('读取 ' + file + ' 结束')
    print('共' + str(len(expand_Dataset)) + '条三元组')
    print('共' + str(sum_valid_row) + '条数据')
    return expand_Dataset


def store_data_to_json(Dataset, file):
    with open(file, 'w', encoding='utf-8') as fout:
        json.dump(Dataset, fout, indent=4, ensure_ascii=False)
    print('数据已存入 ' + file)

