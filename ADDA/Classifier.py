import tensorflow as tf

def LeNet(image,
          is_trainin = False,
          num_clasess=10,
          reuse_private=False,
          private_scope='mnist',
          reuse_shared=False,
          shared_scope='task_model'):
    net = {}