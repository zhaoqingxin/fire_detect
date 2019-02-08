#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time

import numpy as np
import tensorflow as tf

from estimator_model.data.flame_data import train_input_fn, eval_input_fn, predict_input_fn
from db import phoenix

config_path = "config."+os.environ["PYTHON_ENV"]
import importlib
config = importlib.import_module(config_path)

tf.logging.set_verbosity(tf.logging.INFO)

class Estimator_model():
  def __init__(self, model_dir=config.model_dir, image_height=config.image_height, image_width=config.image_width, image_channel=config.image_channel, params=config.params, is_training=False):
    self.model_dir = model_dir
    self.image_width = image_width
    self.image_height = image_height
    self.image_channel = image_channel
    self.params = params
    self.is_training = is_training
    self.classifier = tf.estimator.Estimator(
      model_fn=self.cnn_model_fn, 
      model_dir=self.model_dir,
      params=self.params
    )

    self.tensors_to_log = {
      "probabilities": "softmax_tensor"
    }

    self.logging_hook = tf.train.LoggingTensorHook(
      tensors=self.tensors_to_log, every_n_iter=100
    )

  def evaluate(self,tfrecord_file_path):
    try:
      eval_result = self.classifier.evaluate(
        input_fn=lambda:eval_input_fn(tfrecord_file_path)
      )
      data = (float(eval_result['accuracy']), float(eval_result['loss']), float(eval_result['global_step']))
      phoenix.insert('evaluate',data)
    except Exception as e:
      eval_result={
        'errCode' : 2,
        'massage' : e.message
      }
    finally:
      for i in os.listdir(tfrecord_file_path):
        os.remove(os.path.join(tfrecord_file_path,i))
      os.rmdir(tfrecord_file_path)
      return eval_result
    
    

  def predict(self, file_names, tfrecord_file_path=config.tfrecord_file_path):
    try:
      start = float(time.time())
      predictions = self.classifier.predict(
        input_fn=lambda:predict_input_fn(tfrecord_file_path)
      )
      predict_result={
        'data':[]
      }
      data=[]
      for index,i in enumerate(predictions):
        predict_image = {
          'name' : file_names[index],
          'classify' : int(i['classes']), 
          'probabilities': float(i['probabilities'][int(i['classes'])])
        }
        predict_result['data'].append(predict_image)
        data.append((file_names[index], int(i['classes']), float(i['probabilities'][int(i['classes'])])))
      phoenix.insert('predict',data)
    except Exception as e:
      print(e)
      predict_result = {
        'errCode' : 2,
        'massage' : e.message
      }
    finally:
      for i in os.listdir(tfrecord_file_path):
        os.remove(os.path.join(tfrecord_file_path,i))
      os.rmdir(tfrecord_file_path)
      return predict_result
    

  def train(self,train_image_count,tfrecord_file_path=config.tfrecord_file_path, batch_size=config.batch_size, tfrecord_image_count=config.tfrecord_image_count, tfrecord_file_count=config.tfrecord_file_count, epoch=config.epoch):
    if self.is_training is True:
      print("is training ,return----------------------------------------------------")
      return
    start = int(time.time())
    self.is_training = True
    try:
      steps = 0
      if train_image_count<(train_image_count*tfrecord_file_count):
        steps = train_image_count//batch_size+1
      else:
        steps = (train_image_count*tfrecord_file_count)//batch_size+1
      tfrecord_files = os.listdir(tfrecord_file_path)
      train_tfrecord_file_num=0
      for i in tfrecord_files:
        if 'train' in i:
          train_tfrecord_file_num += 1
      for i in range(epoch):
        for j in range((train_tfrecord_file_num-1)//tfrecord_file_count+1):
          print('epoch: ',i+1,'        ','times: ',j+1)
          self.classifier.train(
            input_fn=lambda:train_input_fn(tfrecord_file_path),
            steps=steps,
            hooks=[self.logging_hook])
          # eval_result = self.classifier.evaluate(
          #   input_fn=lambda:eval_input_fn(tfrecord_file_path)
          # )
      duration = int(time.time())-start
      train_step = (train_tfrecord_file_num//tfrecord_file_count+1)*epoch*steps
      data=(train_image_count, train_step, epoch, duration)
      phoenix.insert('train',data)
    except Exception as e:
      pass
    finally:
      self.is_training = False
      for i in os.listdir(tfrecord_file_path):
        os.remove(os.path.join(tfrecord_file_path,i))
      os.rmdir(tfrecord_file_path)
      print("train---------------------------over")
    

  def cnn_model_fn(self, features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(
      features["image"], 
      [-1, self.image_width, self.image_height, self.image_channel]
    )
    
    height = self.image_height
    width = self.image_width
    channel = self.image_channel

    pool = input_layer
    for c_layer in params['c_layers']:
      # Convolutional Layer
      # Computes features using a kernel_size filter with ReLU activation.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, width, height, channels]
      # Output Tensor Shape: [batch_size, width, height, new channels = c_layer['filters']]
      conv = tf.layers.conv2d(
        inputs=pool,
        filters=c_layer['filters'],
        kernel_size=c_layer['kernel_size'],
        padding="same",
        use_bias=True,
        bias_initializer=tf.random_normal_initializer(stddev=params['stddev']),
        activation=tf.nn.relu
      )

      # Pooling Layer
      # First max pooling layer with a c_layer['pool_size'] filter and stride of c_layer['pool_size']
      # Input Tensor Shape: [batch_size, width, height, channels]
      # Output Tensor Shape: [batch_size, new width, new height, channels]
      pool = tf.layers.max_pooling2d(
        inputs=conv, 
        pool_size=c_layer['pool_size'], 
        strides=c_layer['pool_size'],
      )

      # compute current image size
      height = height/c_layer['strides']
      width = width/c_layer['strides']
      channel = c_layer['filters']

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, height, width, channel]
    # Output Tensor Shape: [batch_size, height * width * channel]
    pool_flat = tf.reshape(pool, [-1, int(height) * int(width) * int(channel)])
    dense = pool_flat
    for d_layer in params['d_layers']:
      # Dense Layer
      # Densely connected layer with d_layer['units'] neurons
      # Input Tensor Shape: [batch_size, width*height*channels]
      # Output Tensor Shape: [batch_size, d_layer['units']]
      dense = tf.layers.dense(inputs=dense, units=d_layer['units'], activation=tf.nn.relu)
      dense_output = dense
      # Add dropout operation; d_layer['dropout'] probability that element will be kept
      if 'dropout' in d_layer.keys():
        dropout = tf.layers.dropout(
          inputs=dense, 
          rate=d_layer['dropout'], 
          training=mode == tf.estimator.ModeKeys.TRAIN
        )
        dense_output = dropout


    # Logits layer
    # Input Tensor Shape: [batch_size, d_layer['units']]
    # Output Tensor Shape: [batch_size, params['classify']]
    logits = tf.layers.dense(inputs=dense_output, units=params['classify'])

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=params['learn_rate'])
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

estimator = Estimator_model()

