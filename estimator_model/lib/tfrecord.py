from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

config_path = "config."+os.environ["PYTHON_ENV"]
import importlib
config = importlib.import_module(config_path)

def create_train_record(data_file_path, train_tfrecord_file_path, eval_tfrecord_file_path, size):
  
  train_writer = tf.python_io.TFRecordWriter(train_tfrecord_file_path)
  eval_writer = tf.python_io.TFRecordWriter(eval_tfrecord_file_path)

  image_count = []
  train_image_count = 0
  eval_image_count = 0
  

  classes = os.listdir(data_file_path)
  
  if ".DS_Store" in classes:
    classes.remove(".DS_Store")
  for index, name in enumerate(classes):
    files = os.listdir(os.path.join(data_file_path,name))
    if ".DS_Store" in classes:
      files.remove(".DS_Store")
    count = 0
    for count, image in enumerate(files):
      imagePath = os.path.join(data_file_path,name,image)
      try:
        img = Image.open(imagePath)
        img = img.resize((size,size))
        img_raw = img.tobytes()
        example = tf.train.Example(
        features=tf.train.Features(feature={
          "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
          'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        if count%10==0:
          eval_writer.write(example.SerializeToString())
          eval_image_count += 1
        else:
          train_writer.write(example.SerializeToString())
          train_image_count += 1
        count += 1
      except:
        print("error:",imagePath)
    image_count.append(count)
  train_writer.close()
  eval_writer.close()
  return train_image_count, eval_image_count, image_count

def create_eval_record(data_file_path, eval_tfrecord_file_path, size):
  
  eval_writer = tf.python_io.TFRecordWriter(eval_tfrecord_file_path)
  image_count = []
  eval_image_count = 0
  classes = os.listdir(data_file_path)
  classes.remove(".DS_Store")
  for index, name in enumerate(classes):
    files = os.listdir(os.path.join(data_file_path,name))
    files.remove(".DS_Store")
    count = 0
    for count, image in enumerate(files):
      count += 1
      imagePath = os.path.join(data_file_path,name,image)
      img = Image.open(imagePath)
      img = img.resize((size,size))
      img_raw = img.tobytes()
      example = tf.train.Example(
      features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
      }))
      eval_writer.write(example.SerializeToString())
      eval_image_count += 1
    image_count.append(count)
  eval_writer.close()
  return eval_image_count, image_count


def _parse_feature(value):
  feature = tf.parse_single_example(
    value,
    features={
      'label': tf.FixedLenFeature([], tf.int64),
      'image':tf.FixedLenFeature([], tf.string)
    }
  )
  feature['image'] = tf.decode_raw(feature['image'],tf.uint8)
  feature['image'] = tf.cast(feature['image'],tf.float32)
  feature['image'] = tf.reshape(feature['image'],[config.image_size,config.image_size,config.image_channel])
  feature['image'] = tf.image.random_flip_left_right(feature['image'])
  feature['image'] = tf.image.random_brightness(feature['image'], max_delta=63)
  feature['image'] = tf.image.random_contrast(feature['image'],lower=0.2, upper=1.8)
  feature['image'] = tf.image.per_image_standardization(feature['image'])
  feature['image'].set_shape([config.image_size,config.image_size,config.image_channel])
  
  label = feature.pop("label")
  return (feature,label)


def read_and_decode(tfrecord_file_path):
  dataset = tf.data.TFRecordDataset(tfrecord_file_path)
  dataset = dataset.map(_parse_feature)
  return dataset