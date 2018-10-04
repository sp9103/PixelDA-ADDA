import logging
import os
import random
from collections import deque

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data_factory import dataset_factory
from common import util
from PixelDA import pixelda_model, pixelda_losses

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

flags.DEFINE_integer('num_preprocessing_threads', 4,
                     'The number of threads used to create the batches.')

flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

flags.DEFINE_integer('snapshot', 5000, '')

flags.DEFINE_integer('iteration', 50000, '')

flags.DEFINE_float('lr', 1e-3, '')

def main(_):
    util.config_logging()

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    seed = random.randrange(2 ** 32 - 2)
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)

    #########################
    # Preprocess the inputs #
    #########################
    source_dataset = dataset_factory.get_dataset(
        FLAGS.source_dataset,
        split_name='train',
        dataset_dir=FLAGS.dataset_dir)
    num_source_classes = source_dataset.num_classes
    source_images, source_labels = dataset_factory.provide_batch(
        FLAGS.source_dataset, 'train', FLAGS.dataset_dir, FLAGS.num_readers,
        32, FLAGS.num_preprocessing_threads)
    source_label = tf.argmax(source_labels['classes'], 1)
    del source_labels['classes']

    target_dataset = dataset_factory.get_dataset(
        FLAGS.target_dataset,
        split_name='test',
        dataset_dir=FLAGS.dataset_dir)
    target_images, _ = dataset_factory.provide_batch(
        FLAGS.target_dataset, 'test', FLAGS.dataset_dir, FLAGS.num_readers,
        32, FLAGS.num_preprocessing_threads)
    num_target_classes = target_dataset.num_classes

    if num_source_classes != num_target_classes:
        raise ValueError(
            'Source and Target datasets must have same number of classes. '
            'Are %d and %d' % (num_source_classes, num_target_classes))

    gen, dis, cls = pixelda_model.create_model(target_images, source_images, num_source_classes, True)

    generator_vars = util.collect_vars('generator')
    discriminator_vars = util.collect_vars('discriminator')
    classfier_vars = util.collect_vars('classifier')

    gen_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
    dis_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
    cls_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'classifier')
    up_op_total = gen_op + dis_op + cls_op + tf.trainable_variables()

    gen_loss = pixelda_losses.g_step_loss(source_images,
                                          source_label,
                                          dis,
                                          cls,
                                          num_source_classes)
    d_loss = pixelda_losses.d_step_loss(dis,
                                        cls,
                                        source_label,
                                        num_source_classes)
    # dis_loss = pixelda_losses.discriminator_loss(dis)
    # cls_loss = pixelda_losses.classification_loss(cls, source_label, num_source_classes)

    learning_rate = tf.train.exponential_decay(
        FLAGS.lr,
        tf.train.get_or_create_global_step(),
        decay_steps=20000,
        decay_rate=0.95,
        staircase=True)

    optimizer = tf.train.AdamOptimizer(
        learning_rate, beta1=0.5)

    dstep_po = dis_op + cls_op
    with tf.control_dependencies(dstep_po):
        dis_step = optimizer.minimize(d_loss, var_list=list(discriminator_vars.values()) + list(classfier_vars.values()))
    with tf.control_dependencies(gen_op):
        gen_step = optimizer.minimize(gen_loss, var_list=list(generator_vars.values()))
    # with tf.control_dependencies(dis_op):
    #    dis_step = optimizer.minimize(dis_loss, var_list=list(discriminator_vars.values()))
    # with tf.control_dependencies(gen_op):
    #    gen_step = optimizer.minimize(gen_loss, var_list=list(generator_vars.values()))
    # with tf.control_dependencies(cls_op):
    #    cls_step = optimizer.minimize(cls_loss, var_list=list(classfier_vars.values()))

    # dis_var_list = dis_op + list(discriminator_vars.values())
    # gen_vars_list = gen_op + list(generator_vars.values())
    # cls_vars_list = cls_op + list(classfier_vars.values())

    # dis_step = optimizer.minimize(dis_loss, var_list=dis_var_list)
    # gen_step = optimizer.minimize(gen_loss, var_list=gen_vars_list)
    # cls_step = optimizer.minimize(cls_loss, var_list=cls_vars_list)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    output_dir = os.path.join('PixelDA/snapshot', 'pixelda')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dis_losses = deque(maxlen=10)
    gen_losses = deque(maxlen=10)
    cls_losses = deque(maxlen=10)
    bar = tqdm(range(FLAGS.iteration))
    bar.set_description('{} (lr: {:.0e})'.format('pixelda', FLAGS.lr))
    bar.refresh()

    #tensorboard
    writer = tf.summary.FileWriter(output_dir, sess.graph)

    display = 10
    stepsize = None
    with slim.queues.QueueRunners(sess):
        for i in bar:
            # d-step
            # dis_loss_val, _ = sess.run([dis_loss, dis_step])
            # dis_losses.append(dis_loss_val)
            # cls_loss_val, _ = sess.run([cls_loss, cls_step])
            # cls_losses.append(cls_loss_val)

            dstep_loss_val, _ = sess.run([d_loss, dis_step])
            dis_losses.append(dstep_loss_val)

            # g-step
            gen_loss_val, _ = sess.run([gen_loss, gen_step])
            gen_losses.append(gen_loss_val)

            if i % display == 0:
                cur_lr = sess.run(learning_rate)
                logging.info('learning rate : {:10.4f}'.format(cur_lr))
                logging.info('{:20} dstep loss: {:10.4f}     (avg: {:10.4f})'
                             '    gen loss: {:10.4f}     (avg: {:10.4f})'
                             .format('Iteration {}:'.format(i),
                                     dstep_loss_val,
                                     np.mean(dis_losses),
                                     gen_loss_val,
                                     np.mean(gen_losses)))

            if (i + 1) % FLAGS.snapshot == 0:
                snapshot_path = saver.save(
                    sess, os.path.join(output_dir, 'pixelda'), global_step=i + 1)
                logging.info('Saved snapshot to {}'.format(snapshot_path))
if __name__ == '__main__':
    tf.app.run()