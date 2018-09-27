import logging
import os
import random
from collections import deque

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_factory import dataset_factory
from common import util
from PixelDA import pixelda_model, pixelda_losses
from tensorflow.python.tools import inspect_checkpoint as chkp

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('source_dataset', 'mnist', 'The name of the source dataset.'
                    ' If hparams="arch=dcgan", this flag is ignored.')

flags.DEFINE_string('target_dataset', 'mnist_m',
                    'The name of the target dataset.')

flags.DEFINE_string('source_split_name', 'train',
                    'Name of the train split for the source.')

flags.DEFINE_string('target_split_name', 'train',
                    'Name of the train split for the target.')

flags.DEFINE_string('dataset_dir', './dataset',
                    'The directory where the datasets can be found.')

flags.DEFINE_integer('num_preprocessing_threads', 1,
                     'The number of threads used to create the batches.')

flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')

def main(_):
    util.config_logging()

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #########################
    # Preprocess the inputs #
    #########################
    source_dataset = dataset_factory.get_dataset(
        FLAGS.source_dataset,
        split_name='test',
        dataset_dir=FLAGS.dataset_dir)
    num_source_classes = source_dataset.num_classes
    source_images, source_labels = dataset_factory.provide_batch(
        FLAGS.source_dataset, 'train', FLAGS.dataset_dir, FLAGS.num_readers,
        1, FLAGS.num_preprocessing_threads)
    source_label = tf.argmax(source_labels['classes'], 1)
    del source_labels['classes']

    target_dataset = dataset_factory.get_dataset(
        FLAGS.target_dataset,
        split_name='test',
        dataset_dir=FLAGS.dataset_dir)
    target_images, target_labels = dataset_factory.provide_batch(
        FLAGS.target_dataset, 'test', FLAGS.dataset_dir, FLAGS.num_readers,
        1, FLAGS.num_preprocessing_threads)
    target_label = tf.argmax(target_labels['classes'], -1)
    del target_labels['classes']
    num_target_classes = target_dataset.num_classes

    if num_source_classes != num_target_classes:
        raise ValueError(
            'Source and Target datasets must have same number of classes. '
            'Are %d and %d' % (num_source_classes, num_target_classes))

    gen, dis, cls = pixelda_model.create_model(target_images, source_images, num_source_classes)

    cls['target_task_logits'] = tf.argmax(cls['target_task_logits'], -1)
    cls['transferred_task_logits'] = tf.argmax(cls['transferred_task_logits'], -1)
    cls['source_task_logits'] = tf.argmax(cls['source_task_logits'], -1)

    # Use the entire split by default
    num_examples = target_dataset.num_samples

    # cls_var_dict = util.collect_vars('classifier')
    # cls_restorer = tf.train.Saver(var_list=cls_var_dict)
    # gen_var_dict = util.collect_vars('generator')
    # gen_restorer = tf.train.Saver(var_list=gen_var_dict)
    restorer = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    output_dir = os.path.join('PixelDA/snapshot', 'pixelda')
    if os.path.isdir(output_dir):
        weights = tf.train.latest_checkpoint(output_dir)
        logging.info('Evaluating {}'.format(weights))

        # print all tensors in checkpoint file
        # chkp.print_tensors_in_checkpoint_file(weights, tensor_name='', all_tensors=True)

        # cls_restorer.restore(sess, weights)
        # gen_restorer.restore(sess, weights)
        restorer.restore(sess, weights)
    else:
        logging.info('Not Found'.format(output_dir))
        return False

    class_correct = np.zeros(num_source_classes, dtype=np.int32)
    class_counts = np.zeros(num_source_classes, dtype=np.int32)

    # classification loss
    with slim.queues.QueueRunners(sess):
        plt.figure()
        for i in range(16):
            np_image, np_gen = sess.run([source_images, gen])
            _, height, width, _ = np_image.shape
            plt.subplot(4, 8, 2 * i + 1)
            plt.imshow(np_image[0])
            plt.title('%d x %d' % (height, width))
            plt.subplot(4, 8, 2 * i + 2)
            np_gen[0] /= 2
            np_gen[0] += 0.5
            plt.imshow(np_gen[0])
            plt.title('%d x %d' % (height, width))
            plt.axis('off')
        plt.show()

        for i in tqdm(range(num_examples)):
            predictions, gt = sess.run([cls['target_task_logits'], target_label])
            class_counts[gt[0]] += 1
            if predictions[0] == gt[0]:
                class_correct[gt[0]] += 1

        logging.info('Class accuracies:')
        logging.info('    ' + util.format_array(class_correct / class_counts))
        logging.info('Overall accuracy:')
        logging.info('    ' + str(np.sum(class_correct) / np.sum(class_counts)))

    return True

if __name__ == '__main__':
    tf.app.run()