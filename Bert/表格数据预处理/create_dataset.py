# -*- encoding: utf-8 -*-
from compact_relation import compact_relation
from read_data_from_xlsx_to_json import read_data_from_xlsx_not_expand, read_data_from_xlsx_expand, store_data_to_json
from create_train_dev_test_from_json import create_train_dev_test_sets
# from combine_two_xlsx_file import combine_file
from expand_from_senence_window import expand_from_sentence_window

if __name__ == '__main__':

    train_dev_test = [0.7, 0.15, 0.15] # 数据集划分比例

    # 将关系缩减为定义的那几类,存储在新文件ds-红楼梦-new.xlsx
    input_file = './data/ds-红楼梦.xlsx'
    output_file = './data/ds-红楼梦-new.xlsx'
    compact_relation(input_file, output_file)

    # read data from xlsx to json
    input_file = './data/ds-红楼梦-new.xlsx'
    output_file = './data/DreamOfTheRedChamber_txt.json'
    data = read_data_from_xlsx_not_expand(input_file)
    store_data_to_json(data, output_file)


    '''
    # expand from sentence_window
    txt_file = './data/DreamOfTheRedChamber_txt.json'
    pic_file = './data/DreamOfTheRedChamber_picture.xlsx'
    output_file = './data/DreamOfTheRedChamber_expand.json'
    name_entity_map_file = './data/name_entity_map_checked.json'
    # 通过句子窗口内存在的sub_obj对扩充
    expand_from_sentence_window(txt_file, pic_file, output_file, name_entity_map_file)
    '''

    # create train/dev/test sets
    input_file = './data/DreamOfTheRedChamber_txt.json'
    train_file = './data/DreamOfTheRedChamber_txt_train.json'
    dev_file = './data/DreamOfTheRedChamber_txt_dev.json'
    test_file = './data/DreamOfTheRedChamber_txt_test.json'
    create_train_dev_test_sets(input_file, train_file, dev_file, test_file, train_dev_test)

