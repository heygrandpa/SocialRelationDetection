import json
import os
import sys
import argparse

def main(args):
	predicate_list = []
	predicate_count = dict()

	anno_info = json.load(open(args.triplet_json_file, 'r', encoding='UTF-8'))
	# print(anno_info)
	for img_name, rels_in_img in anno_info.items():
		for rel in rels_in_img:
			predicate = rel['predicate']
			if predicate not in predicate_list:
				predicate_list.append(predicate)
				predicate_count[predicate] = 1
			else:
				predicate_count[predicate] += 1

	print(len(predicate_list))

	with open(args.predicate_count_file, 'w') as text_file:
		for predicate in predicate_count.keys():
			text_file.write('%s: %d\n' % (predicate, predicate_count[predicate]))

	predicate_count = dict()
	s_o_pairs = json.load(open(args.rel_json_file, 'r', encoding='UTF-8'))
	for s_o_pair in s_o_pairs:
		predicate = s_o_pair['predicate']
		if predicate not in predicate_count.keys():
			predicate_count[predicate] = 1
		else:
			predicate_count[predicate] += 1

	total = sum([predicate_count[predicate] for predicate in predicate_count.keys()])
	print(total)

	with open(args.predicate_count_file2, 'w') as text_file:
		for predicate in predicate_count.keys():
			text_file.write('%s: %d\n' % (predicate, predicate_count[predicate]))


def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--rel_json_file', type=str, help='Json storing valid s_p_o.', default = 'test_valid_triplets.json')
	parser.add_argument('--triplet_json_file', type=str, help='Json storing valid triplets.', default = 'test_annotation.json')
	parser.add_argument('--predicate_count_file', type=str, help='Json storing valid triplets.', default = 'test_predicate.txt')
	parser.add_argument('--predicate_count_file2', type=str, help='Json storing valid triplets.', default = 'test_predicate2.txt')
	return parser.parse_args(argv)
			
if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))