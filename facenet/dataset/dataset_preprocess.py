import json

'''
train_file = 'test_s_p_o_pair.json'
train_s_o_p_pairs = json.load(open(train_file, 'r', encoding='UTF-8'))

rel_list = []

for train_s_o_p_pair in train_s_o_p_pairs:
	rel_dict = dict()
	sbj = train_s_o_p_pair.split('-')[0]
	predicate = train_s_o_p_pair.split('-')[1]
	obj = train_s_o_p_pair.split('-')[2]

	rel_dict['sbj'] = sbj
	rel_dict['obj'] = obj
	rel_dict['predicate'] = predicate
	rel_list.append(rel_dict)

rel_json_file = 'test_valid_triplets.json'

with open(rel_json_file, 'w', encoding='UTF-8') as f:
	json.dump(rel_list, f, ensure_ascii=False)
'''

file = 'test_s_o_pair.json'
test_s_o_pairs = json.load(open(file, 'r', encoding='UTF-8'))

count = 0

for s_o_pair, imgs in test_s_o_pairs.items():
	print('%s: %d' % (s_o_pair, len(imgs)))
	count += len(imgs)

print(count)