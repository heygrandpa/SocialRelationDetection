import json
import random
import math

def create_train_dev_test_sets(input_file, train_file, dev_file, test_file, train_dev_test):
    with open(input_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
        # data为list, 长度为三元组个数
        # 根据三元组划分train/dev/test数据集
        train_triple_num = math.floor(len(data) * train_dev_test[0]) # 三元组数
        dev_triple_num = math.floor(len(data) * train_dev_test[1])
        test_triple_num = len(data) - train_triple_num - dev_triple_num
        train_sample_num = 0
        dev_sample_num = 0
        test_sample_num = 0 # 数据样本数

        ## Train
        # 随机选取train_triple_num个下标
        train_index = random.sample(range(len(data)), train_triple_num)
        train_list = [data[index] for index in train_index]
        # 降序排列train_index
        train_index.sort(reverse=True)
        # 移除已被选择过的数据
        for index in train_index:
            train_sample_num += len(data[index]['txt'])
            data.pop(index)

        ## Dev
        # 随机选取dev_triple_num个下标
        dev_index = random.sample(range(len(data)), dev_triple_num)
        dev_list = [data[index] for index in dev_index]
        # 降序排列dev_index
        dev_index.sort(reverse=True)
        # 移除已被选择过的数据
        for index in dev_index:
            dev_sample_num += len(data[index]['txt'])
            data.pop(index)

        ## Test
        # 余下的均作为test
        test_list = data
        for x in test_list:
            test_sample_num += len(x['txt'])

        # 写入文件
        with open(train_file, 'w', encoding='utf-8') as ftrain, open(dev_file, 'w', encoding='utf-8') as fdev, open(test_file, 'w', encoding='utf-8') as ftest:
            json.dump(train_list, ftrain, indent=4, ensure_ascii=False)
            json.dump(dev_list, fdev, indent=4, ensure_ascii=False)
            json.dump(test_list, ftest, indent=4, ensure_ascii=False)

        print('划分train/dev/test:')
        print('\ttrain: ' + str(train_triple_num) + ' triples,  ' + str(train_sample_num) + ' samples')
        print('\tdev: ' + str(dev_triple_num) + ' triples,  ' + str(dev_sample_num) + ' samples')
        print('\ttest: ' + str(test_triple_num) + ' triples,  ' + str(test_sample_num) + ' samples')