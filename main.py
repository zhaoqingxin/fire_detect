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
import argparse
import random

from estimator_model.lib.tfrecord_fire import create_train_record, create_eval_record, create_predict_record, create_predict_record_from_web, read_and_decode
from estimator_model.estimator import estimator
from estimator_model.data.flame_data import train_input_fn,eval_input_fn
from schedule import schedule
import db

config_path = "config."+os.environ["PYTHON_ENV"]
import importlib
config = importlib.import_module(config_path)

from flaskr import app

parser = argparse.ArgumentParser()
parser.add_argument(
  '--action',
  type=str,
  default='',
  help='if value is train, excute train process'
)



def main():
  app.run('0.0.0.0')

def train():
  tfrecord_file_path = os.path.join(config.tfrecord_file_path, 'train'+str(random.randint(1000,9999))+str(time.time()))
  train_image_count, writers_len = create_train_record(tfrecord_file_path)
  estimator.train(train_image_count, tfrecord_file_path)


FLAGS, unparsed = parser.parse_known_args()

if FLAGS.action == 'train':
  train()
elif FLAGS.action == 'server':
  main()