import os
import csv
import cv2
import sys
import argparse
import numpy as np
import shutil
import xlwt
import json
import xlrd

def change_name(old_name):
	name = old_name
	name = name.replace('少年', '').replace('贾雨村', '贾化').replace('葫芦庙小沙弥', '应天府门子')
	if '袭人' in name and '花袭人' not in name:
		name = name.replace('袭人', '花袭人')
	if '鸳鸯' in name and '金鸳鸯' not in name:
		name = name.replace('鸳鸯', '金鸳鸯')

	return name

def each_result(dir_path):
	result_list = os.listdir(dir_path)
	for i in result_list:
		if '.txt' in i:
			result_list.remove(i)

		if '少年' in i or '袭人' in i:
			pos = result_list.index(i)
			result_list[pos] = change_name(i)

	return result_list


def main(args):
	
	dir_path_list = []
	
	for i in range(args.start_index, args.end_index + 1):
		if i < 10:
			dir_path = '../classified/EP0' + str(i) + '_classified'
		else:
			dir_path = '../classified/EP' + str(i) + '_classified'
		dir_path_list.append(dir_path)

	'''

	# 第一步：统计所有的pair
	all_result = []
	for i in range(len(dir_path_list)):
		dir_path = dir_path_list[i]
		result_list = each_result(dir_path)

		for result in result_list:
			if result not in all_result:
				all_result.append(result)

	# 第二步：遍历 pairs，将 每个文件夹拷贝成 [a, b] 和 [b, a]，对应同一个 pair 的图片合并到同一个文件夹下
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	for pair in all_result:
		print(pair)
		other_pair = pair.split(' ')[1] + ' ' + pair.split(' ')[0]
		pair_path1 = os.path.join(args.output_dir, pair)
		pair_path2 = os.path.join(args.output_dir, other_pair)

		if not os.path.exists(pair_path1):
			os.makedirs(pair_path1)
		if not os.path.exists(pair_path2):
			os.makedirs(pair_path2)

		for dir_path in dir_path_list:
			pair_list = os.listdir(dir_path)
			for dir_pair in pair_list:
				new_dir_pair = change_name(dir_pair)
				if new_dir_pair == pair or new_dir_pair == other_pair:
					dir_pair_path = os.path.join(dir_path, dir_pair)
					img_list = os.listdir(dir_pair_path)
					for img_name in img_list:
						img_path = os.path.join(dir_pair_path, img_name)
						img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1)

						new_img_name = dir_path.split('/')[2].split('_')[0] + '_' + img_name
						new_img_path1 = os.path.join(pair_path1, new_img_name)
						new_img_path2 = os.path.join(pair_path2, new_img_name)
						cv2.imencode('.jpg', img)[1].tofile(new_img_path1)
						cv2.imencode('.jpg', img)[1].tofile(new_img_path2)


	print('All the dirs have been combined!')

	# 第三步：过滤掉不在三元组中的 pair
	if not os.path.exists(args.va_output_dir):
		os.makedirs(args.va_output_dir)

	for pair in all_result:
		sbj = pair.split(' ')[0]
		obj = pair.split(' ')[1]
		other_pair = obj + ' ' + sbj

		csvFile = open('红楼梦.csv','r')
		reader = csv.reader(csvFile)
		next(reader)
		for line in reader:
			# 完备三元组中存在该对pair
			if sbj in line and obj in line:
				old_pair_path1 = os.path.join(args.output_dir, pair)
				old_pair_path2 = os.path.join(args.output_dir, other_pair)

				pair_path1 = os.path.join(args.va_output_dir, pair)
				pair_path2 = os.path.join(args.va_output_dir, other_pair)

				if not os.path.exists(pair_path1):
					shutil.copytree(old_pair_path1, pair_path1)
				if not os.path.exists(pair_path2):
					shutil.copytree(old_pair_path2, pair_path2)

		csvFile.close()

	print('All the dirs have been filtered!')



	# 第四步：统计 pair 的关系
	f = xlwt.Workbook()
	sheet1 = f.add_sheet(u'校对前')
	sheet2 = f.add_sheet(u'校对后')
	
	current_row1 = 0
	current_row2 = 0

	rel_list = []

	valid_pair_list = os.listdir(args.va_output_dir)
	for pair in valid_pair_list:
		sbj = pair.split(' ')[0]
		obj = pair.split(' ')[1]

		rel_dict = dict()
		rel_dict['sbj'] = sbj
		rel_dict['obj'] = obj

		rels_in_pair = []

		csvFile = open('红楼梦.csv','r')
		reader = csv.reader(csvFile)
		next(reader)
		for line in reader:
			if sbj == line[0] and obj == line[1] and line[2] != None:
				rels_in_pair.append(line[2])
		csvFile.close()

		# print('Choose one rel from the following rels:')
		for rel in rels_in_pair:
			sheet1.write(current_row1, 0, sbj)
			sheet1.write(current_row1, 1, obj)
			sheet1.write(current_row1, 2, rel)
			current_row1 += 1

		if len(rels_in_pair) >= 1:
			rel = rels_in_pair[0]
			# if rel != 'relative':
			if True:
				rel_dict['predicate'] = rel
				rel_list.append(rel_dict)

				sheet2.write(current_row2, 0, sbj)
				sheet2.write(current_row2, 1, obj)
				sheet2.write(current_row2, 2, rel)
				current_row2 += 1

	f.save(args.excel_file)


	# 校对完之后进行这一步
	myWorkbook = xlrd.open_workbook(args.excel_file)
	sheet = myWorkbook.sheet_by_name(u'校对后')

	nrows = sheet.nrows
	rel_list = []

	for i in range(nrows):
		rel_dict = dict()
		rel_dict['sbj'] = sheet.cell(i, 0).value
		rel_dict['obj'] = sheet.cell(i, 1).value
		rel_dict['predicate'] = sheet.cell(i, 2).value
		rel_list.append(rel_dict)


	with open(args.rel_json_file, 'w', encoding='UTF-8') as f:
		json.dump(rel_list, f, ensure_ascii=False)

	'''

	rel_list = json.load(open(args.rel_json_file, 'r', encoding='UTF-8'))

	img_count = 0

	# 第五步：合并txt，存储为json文件
	rels_in_imgs = dict()
	for i in range(len(dir_path_list)):
		txt_file_path = os.path.join(dir_path_list[i], 'classified.txt')

		with open(txt_file_path, "r", encoding='UTF-8') as text_file:
			lines = list(text_file)
			label_bbox_dict = dict()

			for line in lines:
				if 'jpg' in line:
					if len(label_bbox_dict.keys()) > 0:
						rels_in_img = construct_rel(label_bbox_dict, rel_list)
						label_bbox_dict = dict()
						if len(rels_in_img) > 0:
							rels_in_imgs[new_img_name] = rels_in_img
							img_count += 1

					img_name = line.split(' ')[-2]
					new_img_name = dir_path_list[i].split('/')[2].split('_')[0] + '_' + img_name

					
				else:
					bbox = line.split(']')[0][1:].split(', ')
					label = line.split(']')[1].split(': ')[1].replace("\n", "")
					label = change_name(label)

					label_bbox_dict[label] = bbox

			# 保存最后一张图片
			rels_in_img = construct_rel(label_bbox_dict, rel_list)
			if len(rels_in_img) > 0:
				rels_in_imgs[new_img_name] = rels_in_img
				img_count += 1

	with open(args.triplet_json_file, 'w', encoding='UTF-8') as f:
		json.dump(rels_in_imgs, f, ensure_ascii=False)

	print(img_count)

	# 第六步：将不同rel的图片文件夹合并到同一个文件夹中
	sbj_obj_file = dict()

	if not os.path.exists(args.img_dir):
		os.makedirs(args.img_dir)
	dir_paths = os.listdir(args.va_output_dir)
	for dir_path in dir_paths:
		sbj = dir_path.split(' ')[0]
		obj = dir_path.split(' ')[1]
		for rel in rel_list:
			if sbj == rel['sbj'] and obj == rel['obj']:
				dir_path = os.path.join(args.va_output_dir, dir_path)
				img_names = os.listdir(dir_path)

				for img_name in img_names:
					img_path = os.path.join(dir_path, img_name)
					img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),-1)

					new_img_path = os.path.join(args.img_dir, img_name)
					cv2.imencode('.jpg', img)[1].tofile(new_img_path)

				s_o_name = sbj + '_' + obj
				sbj_obj_file[s_o_name] = [img_name for img_name in img_names if img_name != 'EP35_0840.jpg' and img_name != 'EP35_0676.jpg']
				if len(sbj_obj_file[s_o_name]) == 0:
					del sbj_obj_file[s_o_name]
				break

	with open(args.s_o_name_json_file, 'w', encoding='UTF-8') as f:
		json.dump(sbj_obj_file, f, ensure_ascii=False)


def construct_rel(label_bbox_dict, rel_list):
	rels_in_img = []
	for sbj in label_bbox_dict.keys():
		for obj in label_bbox_dict.keys():
			for rel in rel_list:
				if sbj == rel['sbj'] and obj == rel['obj']:
					rel_dict = dict()
					rel_dict['predicate'] = rel['predicate']
					
					sbj_dict = dict()
					sbj_dict['category'] = sbj
					sbj_dict['bbox'] = label_bbox_dict[sbj]
					rel_dict['sbj'] = sbj_dict

					obj_dict = dict()
					obj_dict['category'] = obj
					obj_dict['bbox'] = label_bbox_dict[obj]
					rel_dict['obj'] = obj_dict

					rels_in_img.append(rel_dict)
					break

	return rels_in_img

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--start_index', type=int, help='Directory storing classified img.', default = 1)
	parser.add_argument('--end_index', type=int, help='Directory storing classified img.', default = 50)
	parser.add_argument('--output_dir', type=str, help='Directory storing classified img.', default = '../result/all_classified')
	parser.add_argument('--va_output_dir', type=str, help='Directory storing valid classified img.', default = '../result/valid_all_classified')
	parser.add_argument('--img_dir', type=str, help='Directory storing valid classified img.', default = './test_imgs')
	parser.add_argument('--excel_file', type=str, help='Excel storing all rels.', default = './result/all_rels.xls')
	parser.add_argument('--rel_json_file', type=str, help='Json storing valid rels.', default = './test_valid_triplets.json')
	parser.add_argument('--triplet_json_file', type=str, help='Json storing valid triplets.', default = './test_annotation.json')
	parser.add_argument('--s_o_name_json_file', type=str, help='Json storing the image name corresponding to sbj and obj.', default = './test_s_o_pair.json')
	return parser.parse_args(argv)
			
if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))