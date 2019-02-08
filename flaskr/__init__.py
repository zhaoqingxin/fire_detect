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

import numpy as np
import tensorflow as tf

import os
import datetime
import time
import json
import threading
import random

from estimator_model.lib.tfrecord_fire import create_train_record, create_eval_record, create_predict_record, create_predict_record_from_web, read_and_decode
from estimator_model.estimator import estimator
from estimator_model.data.flame_data import train_input_fn,eval_input_fn

config_path = "config."+os.environ["PYTHON_ENV"]
import importlib
config = importlib.import_module(config_path)

from flask import Flask, g, request, url_for, render_template
app = Flask(__name__)

is_training = False

def after_this_request(f):
  if not hasattr(g, 'after_request_callbacks'):
    g.after_request_callbacks = []
  g.after_request_callbacks.append(f)
  return f

@app.after_request
def call_after_request_callbacks(response):
  for callback in getattr(g, 'after_request_callbacks', ()):
    callback(response)
  return response

  
  

class newThread (threading.Thread):   #threading.Thread
  def __init__(self, threadID, fn):
    threading.Thread.__init__(self)
    self.threadID = threadID
    self.fn = fn
  def run(self):                   
    print("Starting " + self.threadID)
    self.fn()
    print("Exiting " + self.threadID)



@app.route("/train", methods=['POST'])
def train():
  tfrecord_file_path = os.path.join(config.tfrecord_file_path, 'train'+str(random.randint(1000,9999))+str(time.time()))
  train_image_count, writers_len = create_train_record(tfrecord_file_path)
  if train_image_count<100:
    result = {
      'errCode' : 1,
      'data':{
        'message':'image count not enough'
      }
    }
    return json.dumps(result)
  thread = newThread(str(time.time()), lambda:estimator.train(train_image_count, tfrecord_file_path))
  thread.start()
  result = {
    'errCode' : 0,
    'data':{
      'train_image_count':train_image_count,
      'message':'train init, you can later search result in database'
    }
  }
  return json.dumps(result)

@app.route("/eval", methods=['POST'])
def evaluate():
  tfrecord_file_path = os.path.join(config.tfrecord_file_path, 'eval'+str(random.randint(1000,9999))+str(time.time()))
  eval_image_count = create_eval_record(tfrecord_file_path)
  if eval_image_count==0:
    result = {
      'errCode' : 1,
      'massage': 'can not found eval dir'
    }
    return json.dumps(result)
  eval_result = estimator.evaluate(tfrecord_file_path)
  if 'errCode' in eval_result.keys():
    result = eval_result
    return json.dumps(result)
  result = {
    'errCode' : 0,
    'data':{
      'accuracy': float(eval_result['accuracy']),
      'loss': float(eval_result['loss']),
      'global_step': float(eval_result['global_step'])
    }
  }
  return json.dumps(result)


@app.route("/predict", methods=['POST'])
def predict():
  start = float(time.time())
  file_paths = []
  file_names = []
  for i in request.files:
    file_paths.append(os.path.join('./upload', request.files[i].filename))
    file_names.append(request.files[i].filename)
    request.files[i].save(os.path.join('./upload', request.files[i].filename))
  if len(file_names)==0:
    result = {
      'errCode' : 1,
      'massage': 'image count equal 0'
    }
    return json.dumps(result)
  tfrecord_file_path = os.path.join(config.tfrecord_file_path, 'predict'+str(random.randint(1000,9999))+str(time.time())+'.tfrecords')
  predict_image_count = create_predict_record_from_web(file_paths,tfrecord_file_path)
  predict_result = estimator.predict(file_names, tfrecord_file_path)
  if 'errCode' in predict_result.keys():
    result = predict_result
    return json.dumps(result)
  result = {
    'errCode':0,
    'data':predict_result['data']
  }
  for i in file_paths:
    os.remove(i)
  end = float(time.time())
  print("duration-----------------------------------",end-start)
  return json.dumps(result)
