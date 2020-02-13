# -*- encoding: utf-8 -*-
import json
from openpyxl import load_workbook
import itertools

def get_text_triples(file):
    # json文件
    with open(file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    # A-relation-B 的格式
    triples = set()
    for x in data:
        triples.add(x['id'])
    return triples

def get_picture_triples(file):
    workbook = load_workbook(file)
    booksheet = workbook.active # 当前活跃的sheet，默认第一个sheet
    # 行数据
    rows = booksheet.rows
    triples = set()
    for row in rows:
        line = [col.value for col in row]
        triples.add('-'.join([line[0], line[2], line[1]]))
    return triples

def combine_triples_from_txt_pic(txt_file, pic_file):
    txt_triples = get_text_triples(txt_file)
    pic_triples = get_picture_triples(pic_file)
    # 取并集
    triples = txt_triples | pic_triples
    triples = list(triples)
    # 抽取对应sub_obj pair & entity
    sub_obj_pairs = []
    entities = set()
    for x in triples:
        temp_split = x.split('-')
        sub_obj_pairs.append('-'.join([temp_split[0], temp_split[2]]))
        entities.add(temp_split[0])
        entities.add(temp_split[2])
    entities = list(entities)
    return triples, sub_obj_pairs, entities

def expand_from_sentence_window(txt_file, pic_file, expand_file, name_entity_map_file):
    triples, sub_obj_pairs, entities = combine_triples_from_txt_pic(txt_file, pic_file)
    # 获取所有的句子窗口，去除重复
    txt_triples_list = []
    with open(txt_file, 'r', encoding='utf-8') as fin, open(name_entity_map_file, 'r', encoding='utf-8') as fin_map, open(expand_file, 'w', encoding='utf-8') as fout:
        data = json.load(fin)
        name_entity_map = json.load(fin_map)
        sentences = set()
        for triple_data in data:
            txt_triples_list.append(triple_data['id'])
            for sen in triple_data['txt']:
                sentences.add(sen['sentence'])
        # 遍历句子窗口
        for sen in sentences:
            entities_in_sen = set()
            for entity in entities:
                if entity in sen:
                    entities_in_sen.add(entity)
            for entity_name in name_entity_map:
                if entity_name in sen:
                    entities_in_sen.add(name_entity_map[entity_name])
            # 笛卡儿积得到在句子窗口中出现的所有sub_obj对
            entities_in_sen = list(entities_in_sen)
            sub_obj_pairs_in_sen = list(itertools.product(entities_in_sen, entities_in_sen))
            # 遍历一个句子中的所有sub_obj pair
            for pair in sub_obj_pairs_in_sen:
                pair = '-'.join(pair)
                if pair in sub_obj_pairs:
                    cor_triple = triples[sub_obj_pairs.index(pair)] # sub_obj对应的triple
                    if cor_triple in txt_triples_list: # triple已存在
                        cor_data = data[txt_triples_list.index(cor_triple)]
                        has_contained = False
                        for x in cor_data['txt']:
                            if x['sentence'] == sen:
                                has_contained = True
                                break
                        if not has_contained: # 但这个句子窗口对于这个triple是新的
                            cor_data['txt'].append({'sentence': sen, 'sentence_marked': sen})
                            data[txt_triples_list.index(cor_triple)] = cor_data
                    else:  # 是新的triple

                        cor_data = {}
                        cor_data['id'] = cor_triple
                        cor_data['sub'] = cor_triple.split('-')[0]
                        cor_data['obj'] = cor_triple.split('-')[2]
                        cor_data['relation'] = cor_triple.split('-')[1]
                        cor_data['txt'] = []
                        cor_data['txt'].append({'sentence': sen, 'sentence_marked': sen})
                        txt_triples_list.append(cor_triple)
                        data.append(cor_data)
        json.dump(data, fout, indent=4, ensure_ascii=False)
