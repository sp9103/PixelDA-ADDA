import logging
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from data_factory import dataset_factory
from ADDA import Classifier
from ADDA import util
import matplotlib.pyplot as plt

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

def format_array(arr):
    return '  '.join(['{:.3f}'.format(x) for x in arr])

def evalutation(session, net, label_batch, num_class, path, name, totalCount, InputImg = None):
    var_dict = util.collect_vars('source_only')
    restorer = tf.train.Saver(var_list=var_dict)
    output_dir = os.path.join(path, name)
    if os.path.isdir(output_dir):
        weights = tf.train.latest_checkpoint(output_dir)
        logging.info('Evaluating {}'.format(weights))
        restorer.restore(session, weights)
    else:
        logging.info('Not Found'.format(output_dir))
        return False

    class_correct = np.zeros(num_class, dtype=np.int32)
    class_counts = np.zeros(num_class, dtype=np.int32)

    with slim.queues.QueueRunners(session):
        plt.figure()
        for i in range(16):
            np_image = session.run(InputImg)
            _, height, width, _ = np_image.shape
            plt.subplot(4, 4, i + 1)
            plt.imshow(np_image[0])
            plt.title('%d x %d' % (height, width))
            plt.axis('off')
        plt.show()

        for i in tqdm(range(totalCount * 2)):
            predictions, gt = session.run([net, label_batch])
            class_counts[gt[0]] += 1
            if predictions[0] == gt[0]:
                class_correct[gt[0]] += 1
        logging.info('Class accuracies:')
        logging.info('    ' + format_array(class_correct / class_counts))
        logging.info('Overall accuracy:')
        logging.info('    ' + str(np.sum(class_correct) / np.sum(class_counts)))

    return True

def main(_):
    util.config_logging()

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

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

    ####################
    # Define the model #
    ####################
    net, layers = Classifier.LeNet(target_images,
                                   False,
                                   num_target_classes,
                                   reuse_private=False,
                                   private_scope='source_only',
                                   reuse_shared=False,
                                   shared_scope='source_only')
    net = tf.argmax(net, -1)

    sess.run(tf.global_variables_initializer())

    #######################################evaluate source only####################################
    evalutation(sess,
                net,
                target_label,
                num_target_classes,
                'ADDA/snapshot',
                'LeNet_mnist',
                target_dataset.num_samples,
                target_images)

    #######################################evaluate adda############################################
    evalutation(sess,
                net,
                target_label,
                num_target_classes,
                'ADDA/snapshot',
                'LeNet_mnist',
                target_dataset.num_samples,
                target_images)

if __name__ == '__main__':
    tf.app.run()