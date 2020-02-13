import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import time
import math
import pickle
import json
import random
import align.detect_face
from six.moves import xrange
from scipy import misc
from six.moves import cPickle as pickle

import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.model_selection import train_test_split

def rel_fc(inputs, is_training=True, mid_dim = 200,
			output_dim = 10, reuse=None, weight_decay=0.0, scope='fc'):

	'''

	# with tf.variable_scope(scope):
	h1 = slim.fully_connected(inputs, mid_dim, activation_fn=None, scope='layer1')
	h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True,is_training=True,scope='bn')
	pro = slim.fully_connected(h2, output_dim, activation_fn=None, scope='layer2')

	return pro

	'''

	batch_norm_params = {
		# Decay for the moving averages.
		'decay': 0.995,
		# epsilon to prevent 0s in variance.
		'epsilon': 0.001,
		# force in-place updates of mean and variance estimates
		'updates_collections': None,
		# Moving averages ends up in the trainable variables collection
		'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
	}

	with slim.arg_scope([slim.fully_connected],
						weights_initializer=slim.initializers.xavier_initializer(), 
						weights_regularizer=slim.l2_regularizer(weight_decay),
						normalizer_fn=slim.batch_norm,
						normalizer_params=batch_norm_params):
		
		dim_in = int(inputs.get_shape()[-1])
		mid_dim = int(dim_in / 2)
		layer1 = slim.fully_connected(inputs, mid_dim, activation_fn=None, scope='layer1')
		pro = slim.fully_connected(layer1, output_dim, activation_fn=None, scope='layer2')

	return pro

def get_old_dataset(imgs_dir, ann_file, s_o_image_file):
	anno_info = json.load(open(ann_file, 'r', encoding='UTF-8'))
	imgs_name = anno_info.keys()
	nrof_imgs = len(imgs_name)

	rel_db = []
	predicate_list = []

	for img_name in imgs_name:
		img_path = os.path.join(imgs_dir, img_name)
		# file_contents = tf.read_file(filename)
		# image = tf.image.decode_image(file_contents, channels=3)

		for rel in anno_info[img_name]:
			if rel['predicate'] not in predicate_list:
				predicate_list.append(rel['predicate'])

			entry = dict(predicate = rel['predicate'],
				sbj_label = rel['sbj']['category'],
				sbj_bbox = rel['sbj']['bbox'],
				obj_label = rel['obj']['category'],
				obj_bbox = rel['obj']['bbox'],
				img_path = img_path)
			rel_db.append(entry)

	rel_db_size = len(rel_db)
	print("%d rel entries" % (rel_db_size))

	return rel_db, predicate_list

def get_last_dataset(imgs_dir, ann_file, s_o_image_file):
	anno_info = json.load(open(ann_file, 'r', encoding='UTF-8'))
	s_o_image = json.load(open(s_o_image_file, 'r', encoding='UTF-8'))
	imgs_name = anno_info.keys()
	s_o_pairs = s_o_image.keys()
	nrof_imgs = len(imgs_name)

	rel_db = dict()
	predicate_list = []

	for s_o_pair in s_o_pairs:
		img_names = s_o_image[s_o_pair]
		s_o_db = []
		for img_name in imgs_name:
			img_path = os.path.join(imgs_dir, img_name)
			for rel in anno_info[img_name]:
				sbj = s_o_pair.split('_')[0]
				obj = s_o_pair.split('_')[1]
				if rel['sbj']['category'] == sbj and rel['obj']['category'] == obj:
					if rel['predicate'] not in predicate_list:
						predicate_list.append(rel['predicate'])

					entry = dict(predicate = rel['predicate'],
						sbj_label = rel['sbj']['category'],
						sbj_bbox = rel['sbj']['bbox'],
						obj_label = rel['obj']['category'],
						obj_bbox = rel['obj']['bbox'],
						img_path = img_path)
					s_o_db.append(entry)

		rel_db[s_o_pair] = s_o_db

	rel_db_size = len(rel_db.keys())
	s_o_pairs = [s_o_pair for s_o_pair in s_o_pairs]
	print("%d sbj_obj_pairs and %d rel entries" % (rel_db_size, sum(len(rel_db[s_o_pair]) for s_o_pair in rel_db.keys())))

	return rel_db, predicate_list, s_o_pairs

def get_dataset(imgs_dir, ann_file, s_o_image_file):
	anno_info = json.load(open(ann_file, 'r', encoding='UTF-8'))
	s_o_image = json.load(open(s_o_image_file, 'r', encoding='UTF-8'))
	imgs_name = anno_info.keys()
	s_o_pairs = s_o_image.keys()
	nrof_imgs = len(imgs_name)

	rel_db = []
	predicate_list = []

	for s_o_pair in s_o_pairs:
		img_names = s_o_image[s_o_pair]
		for img_name in imgs_name:
			img_path = os.path.join(imgs_dir, img_name)
			for rel in anno_info[img_name]:
				sbj = s_o_pair.split('_')[0]
				obj = s_o_pair.split('_')[1]
				if rel['sbj']['category'] == sbj and rel['obj']['category'] == obj:
					if rel['predicate'] not in predicate_list:
						predicate_list.append(rel['predicate'])

					entry = dict(predicate = rel['predicate'],
						sbj_label = rel['sbj']['category'],
						sbj_bbox = rel['sbj']['bbox'],
						obj_label = rel['obj']['category'],
						obj_bbox = rel['obj']['bbox'],
						img_path = img_path)
					rel_db.append(entry)

	num_entries = len(rel_db)
	s_o_pairs = [s_o_pair for s_o_pair in s_o_pairs]
	num_sbj_obj_pairs = len(s_o_pairs)
	print("%d sbj_obj_pairs and %d rel entries" % (num_sbj_obj_pairs, num_entries))

	return rel_db, predicate_list, s_o_pairs

def old_train(sess, epoch, dataset, embeddings, images_placeholder, phase_train_placeholder, embs_placeholder,
			label_placeholder, learning_rate_placeholder, embedding_size, batch_size, learning_rate,
			image_size, predictions, outputs, loss, train_op, predicate_list):
	nrof_examples = sum(len(value) for value in dataset.values())
	nrof_batches = int(np.ceil(nrof_examples / batch_size))
	nrof_labels = len(predicate_list)
	nrof_correct = 0

	s_o_pairs = [key for key in dataset.keys()]
	s_o_pair_index = 0
	in_s_o_pair_index = 0

	loss_list = np.zeros(nrof_batches, )

	for i in range(nrof_batches):
		start_time = time.time()
		# start = i * batch_size
		# end = min((i + 1) * batch_size, nrof_examples)
		# batch_data = dataset[start:end]

		current_batch_size = min(batch_size, nrof_examples - i * batch_size)
		batch_data = [None for i in range(current_batch_size)]
		for j in range(current_batch_size):
			if len(dataset[s_o_pairs[s_o_pair_index]]) == 0:
				print(s_o_pairs[s_o_pair_index])

			batch_data[j] = dataset[s_o_pairs[s_o_pair_index]][in_s_o_pair_index]
			in_s_o_pair_index += 1
			if in_s_o_pair_index >= len(dataset[s_o_pairs[s_o_pair_index]]):
				s_o_pair_index += 1
				in_s_o_pair_index = 0

		sbj_img_list = []
		obj_img_list = []
		rel_list = np.zeros((current_batch_size, nrof_labels), dtype=np.float32)

		for j in range(current_batch_size):
			entry = batch_data[j]
			img_path = entry['img_path']
			img = misc.imread(os.path.expanduser(img_path))

			sbj_bbox = entry['sbj_bbox']
			sbj_bbox = [int(num) for num in sbj_bbox]
			sbj_cropped = img[sbj_bbox[1]:sbj_bbox[3],sbj_bbox[0]:sbj_bbox[2],:]
			aligned = misc.imresize(sbj_cropped, (image_size, image_size), interp='bilinear')
			prewhitened = facenet.prewhiten(aligned)
			sbj_img_list.append(prewhitened)

			obj_bbox = entry['obj_bbox']
			obj_bbox = [int(num) for num in obj_bbox]
			obj_cropped = img[obj_bbox[1]:obj_bbox[3], obj_bbox[0]:obj_bbox[2],:]
			aligned = misc.imresize(obj_cropped, (image_size, image_size), interp='bilinear')
			prewhitened = facenet.prewhiten(aligned)
			obj_img_list.append(prewhitened)

			predicate = entry['predicate']
			rel_list[j][predicate_list.index(predicate)] = 1

		sbj_images = np.stack(sbj_img_list)
		obj_images = np.stack(obj_img_list)

		# 计算 facenet embdding 作为 feature
		sbj_feed_dict = {images_placeholder: sbj_images , phase_train_placeholder: False}
		sbj_emb = sess.run(embeddings, feed_dict=sbj_feed_dict) # shape: (100, 512)
		obj_feed_dict = {images_placeholder: obj_images , phase_train_placeholder: False}
		obj_emb = sess.run(embeddings, feed_dict=obj_feed_dict) # shape: (100, 512)
		delta_emb = sbj_emb - obj_emb
		# product_emb = sbj_emb * obj_emb

		emb = np.concatenate((sbj_emb, obj_emb, delta_emb), axis=1) # shape: (100, 1536)
		# emb = np.concatenate((sbj_emb, obj_emb, product_emb), axis=1) # shape: (100, 1536)

		# 预测分类
		feed_dict = {embs_placeholder: emb, label_placeholder: rel_list, learning_rate_placeholder: learning_rate}
		pred, output, err, _ = sess.run([predictions, outputs, loss, train_op], feed_dict=feed_dict)

		loss_list[i] = err  # 损失函数

		# 准确率
		pred = np.squeeze(np.array(pred))
		pred_label = np.argmax(pred, axis=1)
		rel_list = np.argmax(rel_list, axis=1)

		batch_correct = np.where(pred_label == rel_list)[0].shape[0]
		nrof_correct += batch_correct

		duration = time.time() - start_time
		print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\t Accuracy: %2.3f' %
				(epoch, i + 1, nrof_batches, duration, err, batch_correct / current_batch_size))

	err = np.mean(loss_list)
	return err, nrof_correct / nrof_examples

def train(sess, epoch, dataset, embeddings, images_placeholder, phase_train_placeholder, embs_placeholder,
			label_placeholder, learning_rate_placeholder, embedding_size, batch_size, learning_rate,
			image_size, predictions, outputs, loss, train_op, predicate_list):
	nrof_examples = len(dataset)
	nrof_labels = len(predicate_list)
	nrof_correct = 0
	nrof_batches = int(np.ceil(nrof_examples / batch_size))

	# 打乱数据集
	index = [i for i in range(len(dataset))]
	random.shuffle(index)
	dataset = np.array(dataset)[index].tolist()

	left_index = 0

	loss_list = np.zeros(nrof_batches, )

	for i in range(nrof_batches):
		start_time = time.time()
		# start = i * batch_size
		# end = min((i + 1) * batch_size, nrof_examples)
		# batch_data = dataset[start:end]

		current_batch_size = min(batch_size, nrof_examples - i * batch_size)
		batch_data = dataset[left_index: left_index + current_batch_size]

		sbj_img_list = []
		obj_img_list = []
		rel_list = np.zeros((current_batch_size, nrof_labels), dtype=np.float32)

		for j in range(current_batch_size):
			entry = batch_data[j]
			img_path = entry['img_path']
			img = misc.imread(os.path.expanduser(img_path))

			sbj_bbox = entry['sbj_bbox']
			sbj_bbox = [int(num) for num in sbj_bbox]
			sbj_cropped = img[sbj_bbox[1]:sbj_bbox[3],sbj_bbox[0]:sbj_bbox[2],:]
			aligned = misc.imresize(sbj_cropped, (image_size, image_size), interp='bilinear')
			prewhitened = facenet.prewhiten(aligned)
			sbj_img_list.append(prewhitened)

			obj_bbox = entry['obj_bbox']
			obj_bbox = [int(num) for num in obj_bbox]
			obj_cropped = img[obj_bbox[1]:obj_bbox[3], obj_bbox[0]:obj_bbox[2],:]
			aligned = misc.imresize(obj_cropped, (image_size, image_size), interp='bilinear')
			prewhitened = facenet.prewhiten(aligned)
			obj_img_list.append(prewhitened)

			predicate = entry['predicate']
			rel_list[j][predicate_list.index(predicate)] = 1

		sbj_images = np.stack(sbj_img_list)
		obj_images = np.stack(obj_img_list)

		# 计算 facenet embdding 作为 feature
		sbj_feed_dict = {images_placeholder: sbj_images , phase_train_placeholder: False}
		sbj_emb = sess.run(embeddings, feed_dict=sbj_feed_dict) # shape: (100, 512)
		obj_feed_dict = {images_placeholder: obj_images , phase_train_placeholder: False}
		obj_emb = sess.run(embeddings, feed_dict=obj_feed_dict) # shape: (100, 512)
		# delta_emb = sbj_emb - obj_emb
		product_emb = sbj_emb * obj_emb

		# emb = np.concatenate((sbj_emb, obj_emb, delta_emb), axis=1) # shape: (100, 1536)
		emb = np.concatenate((sbj_emb, obj_emb, product_emb), axis=1) # shape: (100, 1536)

		# 预测分类
		feed_dict = {embs_placeholder: emb, label_placeholder: rel_list, learning_rate_placeholder: learning_rate}
		pred, output, err, _ = sess.run([predictions, outputs, loss, train_op], feed_dict=feed_dict)

		loss_list[i] = err  # 损失函数

		# 准确率
		pred = np.squeeze(np.array(pred))
		pred_label = np.argmax(pred, axis=1)
		rel_list = np.argmax(rel_list, axis=1)

		batch_correct = np.where(pred_label == rel_list)[0].shape[0]
		nrof_correct += batch_correct

		duration = time.time() - start_time
		print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\t Accuracy: %2.3f' %
				(epoch, i + 1, nrof_batches, duration, err, batch_correct / current_batch_size))

	err = np.mean(loss_list)
	return err, nrof_correct / nrof_examples

def evaluate(sess, dataset, embeddings, images_placeholder, phase_train_placeholder, embs_placeholder,
			label_placeholder, learning_rate_placeholder, embedding_size, batch_size, image_size, predictions, predicate_list):
	nrof_examples = len(dataset)
	nrof_batches = int(np.ceil(nrof_examples / batch_size))
	nrof_labels = len(predicate_list)
	nrof_correct = 0

	left_index = 0
	nrof_correct = 0

	start_time = time.time()

	for i in range(nrof_batches):
		current_batch_size = min(batch_size, nrof_examples - i * batch_size)
		batch_data = dataset[left_index: left_index + current_batch_size]

		sbj_img_list = []
		obj_img_list = []
		rel_list = np.zeros((current_batch_size, ), dtype=np.float32)

		for j in range(current_batch_size):
			entry = batch_data[j]
			img_path = entry['img_path']
			img = misc.imread(os.path.expanduser(img_path))

			sbj_bbox = entry['sbj_bbox']
			sbj_bbox = [int(num) for num in sbj_bbox]
			sbj_cropped = img[sbj_bbox[1]:sbj_bbox[3],sbj_bbox[0]:sbj_bbox[2],:]
			aligned = misc.imresize(sbj_cropped, (image_size, image_size), interp='bilinear')
			prewhitened = facenet.prewhiten(aligned)
			sbj_img_list.append(prewhitened)

			obj_bbox = entry['obj_bbox']
			obj_bbox = [int(num) for num in obj_bbox]
			obj_cropped = img[obj_bbox[1]:obj_bbox[3], obj_bbox[0]:obj_bbox[2],:]
			aligned = misc.imresize(obj_cropped, (image_size, image_size), interp='bilinear')
			prewhitened = facenet.prewhiten(aligned)
			obj_img_list.append(prewhitened)

			predicate = entry['predicate']
			rel_list[j] = predicate_list.index(predicate)

		sbj_images = np.stack(sbj_img_list)
		obj_images = np.stack(obj_img_list)

		# 计算 facenet embdding 作为 feature
		sbj_feed_dict = {images_placeholder: sbj_images , phase_train_placeholder: False}
		sbj_emb = sess.run(embeddings, feed_dict=sbj_feed_dict) # shape: (100, 512)
		obj_feed_dict = {images_placeholder: obj_images , phase_train_placeholder: False}
		obj_emb = sess.run(embeddings, feed_dict=obj_feed_dict) # shape: (100, 512)
		# delta_emb = sbj_emb - obj_emb
		product_emb = sbj_emb * obj_emb

		# emb = np.concatenate((sbj_emb, obj_emb, delta_emb), axis=1) # shape: (100, 1536)
		emb = np.concatenate((sbj_emb, obj_emb, product_emb), axis=1)

		# 预测分类
		feed_dict = {embs_placeholder: emb, learning_rate_placeholder: 0.0, phase_train_placeholder: False}
		pred = sess.run([predictions], feed_dict=feed_dict)
		pred = np.squeeze(np.array(pred))
		pred_label = np.argmax(pred, axis=1)

		print(pred_label)

		nrof_correct += np.where(pred_label == rel_list)[0].shape[0]

	duration = time.time() - start_time
	print('Accuracy: %2.3f\t Time %.3f' %
			(nrof_correct / nrof_examples, duration))

	return nrof_correct / nrof_examples


def main(args):
	if not os.path.isdir(args.model_dir):  # Create the model directory if it doesn't exist
		os.makedirs(args.model_dir)

	np.random.seed(seed=args.seed)

	'''
	rel_db, predicate_list, s_o_pairs = get_dataset(args.imgs_dir, args.ann_file, args.s_o_img_file)
	train_s_o_pairs, test_s_o_pairs = train_test_split(s_o_pairs, train_size=0.8, test_size=0.2)
	train_dataset = {key: value for key, value in rel_db.items() if key in train_s_o_pairs}
	test_dataset = {key: value for key, value in rel_db.items() if key in test_s_o_pairs}
	'''

	imgs_dir = os.path.join(args.dataset_dir, 'imgs')
	train_ann_file = os.path.join(args.dataset_dir, 'train_annotation.json')
	train_s_o_img_file = os.path.join(args.dataset_dir, 'train_s_o_pair.json')
	train_dataset, predicate_list, train_s_o_pairs = get_dataset(imgs_dir, train_ann_file, train_s_o_img_file)

	val_ann_file = os.path.join(args.dataset_dir, 'val_annotation.json')
	val_s_o_img_file = os.path.join(args.dataset_dir, 'val_s_o_pair.json')
	val_dataset, _, val_s_o_pairs = get_dataset(imgs_dir, val_ann_file, val_s_o_img_file)

	test_ann_file = os.path.join(args.dataset_dir, 'test_annotation.json')
	test_s_o_img_file = os.path.join(args.dataset_dir, 'test_s_o_pair.json')
	test_dataset, _, test_s_o_pairs = get_dataset(imgs_dir, test_ann_file, test_s_o_img_file)


	nrof_labels = len(predicate_list)
	print(predicate_list)

	'''

	print("%d relations and %d rel entries in train dataset" % (len(train_s_o_pairs),
		sum([len(train_dataset[train_s_o]) for train_s_o in train_s_o_pairs])))
	print("%d relations and %d rel entries in val dataset" % (len(val_s_o_pairs),
		sum([len(val_dataset[val_s_o]) for val_s_o in val_s_o_pairs])))
	print("%d relations and %d rel entries in test dataset" % (len(test_s_o_pairs),
		sum([len(test_dataset[test_s_o]) for test_s_o in test_s_o_pairs])))

	'''

	print("%d relations and %d rel entries in train dataset" % (len(train_s_o_pairs), len(train_dataset)))
	print("%d relations and %d rel entries in val dataset" % (len(val_s_o_pairs), len(val_dataset)))
	print("%d relations and %d rel entries in test dataset" % (len(test_s_o_pairs), len(test_dataset)))

	with tf.Graph().as_default():
		tf.set_random_seed(args.seed)
		global_step = tf.Variable(0, trainable=False)

		# Start running operations on the Graph.
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

		with sess.as_default():
			# Load facenet model
			facenet.load_model(args.model)
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

			# classifier
			embedding_size = embeddings.get_shape()[1]
			embs_placeholder = tf.placeholder(tf.float32, [None, embedding_size * 3])
			label_placeholder = tf.placeholder(tf.float32, [None, nrof_labels]) # 用 one-hot 表示
			learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

			learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
				args.learning_rate_decay_epochs * args.batch_size, args.learning_rate_decay_factor, staircase = False)

			outputs = rel_fc(inputs = embs_placeholder, is_training=True, output_dim = nrof_labels)
			predictions = tf.squeeze(tf.nn.softmax(outputs))
			# loss = slim.losses.softmax_cross_entropy(predictions, label_placeholder)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = predictions, labels = label_placeholder))

			# Optimizer
			opt = tf.train.AdamOptimizer(learning_rate)
			train_op = opt.minimize(loss)

			# Create a saver
			# print(tf.trainable_variables())
			# assert False
			# 这里 facenet 好像没有 freeze 住
			saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)  # max_to_keep: 保留文件的最大数量

			# init = tf.global_variables_initializer()
			# sess.run(init)

			# 初始化未初始化的变量
			global_vars = tf.global_variables()
			is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
			not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
			# print ([str(i.name) for i in not_initialized_vars]) # only for testing
			if len(not_initialized_vars):
				sess.run(tf.variables_initializer(not_initialized_vars))

			if args.is_train:
				# Training Loop
				err_list = np.zeros(args.max_nrof_epochs, )
				acc_list = np.zeros(args.max_nrof_epochs, )
				max_acc = 0
				epoch = 0

				while epoch < args.max_nrof_epochs:
					err_list[epoch], acc_list[epoch] = train(sess, epoch, train_dataset, embeddings, images_placeholder, phase_train_placeholder, embs_placeholder,
						label_placeholder, learning_rate_placeholder, embedding_size, args.batch_size, args.learning_rate,
						args.image_size, predictions, outputs, loss, train_op, predicate_list)
					acc = evaluate(sess, val_dataset, embeddings, images_placeholder, phase_train_placeholder, embs_placeholder,
						label_placeholder, learning_rate_placeholder, embedding_size, args.batch_size, args.image_size, predictions, predicate_list)
					if acc > max_acc:
						max_acc = acc
						checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
						saver.save(sess, checkpoint_path, global_step=epoch + 1)

					epoch += 1

				print(err_list)
				print(acc_list)

				acc_loss = dict(loss = err_list, acc = acc_list)
				with open(args.Acc_Loss_Result_File, 'wb') as f:
					pickle.dump(acc_loss, f, pickle.HIGHEST_PROTOCOL)

			else:
				model_file = tf.train.latest_checkpoint(args.model_dir)
				saver.restore(sess, model_file)
				# Evaluate
				evaluate(sess, test_dataset, embeddings, images_placeholder, phase_train_placeholder, embs_placeholder,
						label_placeholder, learning_rate_placeholder, embedding_size, args.batch_size, args.image_size, predictions, predicate_list)

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	'''
	parser.add_argument('--imgs_dir', type=str, help='Directory storing img.',
		default = '/home/lab/zmr/facenet/imgs/result/imgs')
	parser.add_argument('--ann_file', type=str, help='Json storing valid rels.',
		default = '/home/lab/zmr/facenet/imgs/result/annotation.json')
	parser.add_argument('--s_o_img_file', type=str, help='Json storing the image name corresponding to sbj and obj.',
		default = '/home/lab/zmr/facenet/imgs/result/s_o_pair.json')
	'''
	parser.add_argument('--dataset_dir', type=str, help='Directory storing dataset.',
		default = '/home/lab/zmr/facenet/imgs/dataset')
	parser.add_argument('--model', type=str,
		help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
		# default='/home/lab/zmr/facenet/20180402-114759')
		default='./20180402-114759')
	parser.add_argument('--model_dir', type=str,
		help='The directory containing the meta_file and ckpt_file or a model protobuf (.pb) file of fully connected layer',
		# default='/home/lab/zmr/facenet/imgs/model_product/model-01-09-20')
		default='./model_product/model-01-09-20')
	parser.add_argument('--gpu_memory_fraction', type=float,
		help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.7)
	parser.add_argument('--nrof_classes', type=int,
		help='The number of relation classes', default=10)
	parser.add_argument('--batch_size', type=int, help='batch size', default = 1000)
	parser.add_argument('--max_nrof_epochs', type=int, help='Number of epochs to run.', default=50)
	parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
	parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=0.1)
	parser.add_argument('--learning_rate', type=float,
		help='Initial learning rate. ', default=0.01)
	parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=5)
	parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
	parser.add_argument('--Acc_Loss_Result_File', type=str, help='The Json File that stores Accuracy and Loss result',
		default = '/home/lab/zmr/facenet/imgs/product_acc_loss_s_o_pair-01-09-20.pkl')
	parser.add_argument('--is_train', action='store_true', default=False)
	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))