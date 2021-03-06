import logging
import logging.config
import os.path
from collections import OrderedDict
from tensorflow.python import pywrap_tensorflow

import yaml

import tensorflow as tf
import numpy as np
from tqdm import tqdm

class TqdmHandler(logging.StreamHandler):

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            # var_name = remove_first_scope(var.op.name)
            var_name = var.op.name
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            # var_name = remove_first_scope(var.op.name)
            var_name = var.op.name
            if var_name.startswith(end):
                end = i
                break
    for var in vars[start:end]:
        # var_name = remove_first_scope(var.op.name)
        var_name = var.op.name
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict

def config_logging(logfile=None):
    path = os.path.join(os.path.dirname(__file__), 'logging.yml')
    with open(path, 'r') as f:
        config = yaml.load(f.read())
    if logfile is None:
        del config['handlers']['file_handler']
        del config['root']['handlers'][-1]
    else:
        config['handlers']['file_handler']['filename'] = logfile
    logging.config.dictConfig(config)

def format_array(arr):
    return '  '.join(['{:.3f}'.format(x) for x in arr])

def ConvertKey(src, str):
    returnDict = OrderedDict()

    for dstkey, dstval in src.items():
        keylist = dstkey.split('/')
        newkey = str
        for i in range(len(keylist) - 1):
            newkey += '/'
            newkey += keylist[i + 1]
        returnDict[newkey] = dstval
    return returnDict

def copyKeySet(src, dst):
    returnDict = OrderedDict()

    for dstkey, dstval in dst.items():
        dst_scope_set = parseScope(dstkey)
        for srckey, srcval in src.items():
            src_scope_set = parseScope(srckey)

            if src_scope_set.count('task_model') != 0:
                src_scope_set.remove('task_model')
            elif src_scope_set.count('transferred_task_classifier') != 0:
                src_scope_set.remove('transferred_task_classifier')
            else:
                continue

            dstlayername = dst_scope_set[0]
            srclayername = src_scope_set[0]
            dstlayerType = dst_scope_set[1]
            srclayerType = src_scope_set[1]

            if dstlayername == srclayername\
                and dstlayerType == srclayerType:
                returnDict[srckey] = dstval
                break

    return returnDict

def parseScope(str):
    keylist = str.split('/')

    if keylist.count('classifier') != 0:
        keylist.remove('classifier')
    if keylist.count('target') != 0:
        keylist.remove('target')

    return keylist

def restoreNetforEval(var_dict, path, name, session):
    restorer = tf.train.Saver(var_list=var_dict)
    output_dir = os.path.join(path, name)
    if os.path.isdir(output_dir):
        weights = tf.train.latest_checkpoint(output_dir)
        printSavedInfo(weights)
        restorer.restore(session, weights)
    else:
        logging.info('Not Found'.format(output_dir))
        return False

def printSavedInfo(path, printValue = False):
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    for key in var_to_shape_map:
        print("tensor_name: ", key)
        if printValue:
            print(reader.get_tensor(key))  # Remove this is you want to print only variable names

def evalutation(session, net, label_batch, num_class, path, name, totalCount, var_dict, InputImg = None):
    if restoreNetforEval(var_dict, path, name, session) == False:
        return False

    class_correct = np.zeros(num_class, dtype=np.int32)
    class_counts = np.zeros(num_class, dtype=np.int32)

    # plt.figure()
    # for i in range(16):
    #     np_image = session.run(InputImg)
    #     _, height, width, _ = np_image.shape
    #     plt.subplot(4, 4, i + 1)
    #     plt.imshow(np_image[0])
    #     plt.title('%d x %d' % (height, width))
    #     plt.axis('off')
    # plt.show()

    for i in tqdm(range(totalCount)):
        predictions, gt = session.run([net, label_batch])
        class_counts[gt[0]] += 1
        if predictions[0] == gt[0]:
            class_correct[gt[0]] += 1
    logging.info('Class accuracies:')
    logging.info('    ' + format_array(class_correct / class_counts))
    logging.info('Overall accuracy:')
    logging.info('    ' + str(np.sum(class_correct) / np.sum(class_counts)))

    return True