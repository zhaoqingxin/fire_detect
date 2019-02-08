from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from estimator_model.lib.tfrecord_fire import read_and_decode, predict_read_and_decode

config_path = "config."+os.environ["PYTHON_ENV"]
import importlib
config = importlib.import_module(config_path)

def train_input_fn(tfrecord_file_path=config.tfrecord_file_path, batch_size=config.batch_size, tfrecord_image_count=config.tfrecord_image_count, tfrecord_file_count=config.tfrecord_file_count):
  dataset = read_and_decode('train',tfrecord_file_path)
  dataset = dataset.shuffle(tfrecord_image_count*tfrecord_file_count).repeat().batch(batch_size)
  return dataset
    
def eval_input_fn(tfrecord_file_path=config.tfrecord_file_path,batch_size=config.batch_size):
  dataset = read_and_decode('eval', tfrecord_file_path)
  dataset = dataset.batch(batch_size)
  return dataset

def predict_input_fn(tfrecord_file_path=config.tfrecord_file_path,batch_size=config.batch_size):
  dataset = predict_read_and_decode(tfrecord_file_path)
  dataset = dataset.batch(batch_size)
  return dataset


#   dataset = train_input_fn(config.train_tfrecord_file_path,config.batch_size,10000)
#   dataset = dataset.make_initializable_iterator()
  
#   with tf.Session() as sess:
#     sess.run(dataset.initializer)
    
#     for i in range(10):
#       x,y = dataset.get_next()
#       print(y.eval())
#     return