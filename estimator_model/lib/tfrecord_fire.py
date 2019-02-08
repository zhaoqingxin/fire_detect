from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

config_path = "config."+os.environ["PYTHON_ENV"]
import importlib
config = importlib.import_module(config_path)

def create_train_record(tfrecord_file_path=config.tfrecord_file_path, data_file_path=config.data_file_path, image_width=config.image_width, image_height=config.image_height, tfrecord_image_count=config.tfrecord_image_count):
  print("create_train_record--------------------start")
  data_file_path = os.path.join(data_file_path,"train")
  if not os.path.exists(data_file_path):
    os.makedirs(data_file_path)

  if not os.path.exists(tfrecord_file_path):
    os.makedirs(tfrecord_file_path)

  classify = os.listdir(data_file_path)
  # disgusting mac
  if ".DS_Store" in classify:
    classify.remove(".DS_Store")
  image_count = 0
  for image_dir in classify:
    image_dir_path = os.path.join(data_file_path,image_dir)
    image_count += len(os.listdir(image_dir_path))
  writers = []
  if image_count<tfrecord_image_count:
    writers.append(tf.python_io.TFRecordWriter(os.path.join(tfrecord_file_path, "train0.tfrecords")))
  else:
    for i in range(image_count//tfrecord_image_count+1):
      writers.append(tf.python_io.TFRecordWriter(os.path.join(tfrecord_file_path, "train"+str(i)+".tfrecords")))
  # writers.append(tf.python_io.TFRecordWriter(os.path.join(tfrecord_file_path,"eval.tfrecords")))

  writers_len = len(writers)
  train_image_count = 0
  # eval_image_count = 0

  for image_dir in classify:
    image_dir_path = os.path.join(data_file_path,image_dir)
    images = os.listdir(image_dir_path)
    if ".DS_Store" in images:
      images.remove(".DS_Store")
    for index, image in enumerate(images):
      image_path = os.path.join(image_dir_path,image)
      try:
        img = Image.open(image_path)
        # w,h = img.size
        # img = img.crop([w//2-32,h//2-32,w//2+32,h//2+32])
        img = img.resize((image_width,image_height))
        img_raw = img.tobytes()
        example = tf.train.Example(
        features=tf.train.Features(feature={
          "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(image_dir)])),
          'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        i = index%writers_len
        writers[i].write(example.SerializeToString())
        train_image_count += 1
        # if i==(writers_len-1):
        #   eval_image_count += 1
        # else:
        #   train_image_count += 1
      except:
        print("error:",image_path)
  
  for writer in writers:
    writer.close()
  # return train_image_count, eval_image_count, writers_len
  return train_image_count, writers_len

def create_eval_record(tfrecord_file_path=config.tfrecord_file_path, data_file_path=config.data_file_path, image_width=config.image_width,image_height=config.image_height):
  print("create_eval_record--------------------start")

  data_file_path = os.path.join(data_file_path,"eval")
  if not os.path.exists(data_file_path):
    os.makedirs(data_file_path)
  if not os.path.exists(tfrecord_file_path):
    os.makedirs(tfrecord_file_path)
  # fire_dir = os.path.join(data_file_path,"fire")
  # normal_dir = os.path.join(data_file_path,"normal")
  # if not os.path.exists(fire_dir) or not os.path.exists(normal_dir):
  #   return 0
  # fire_imgs = os.listdir(fire_dir)
  # normal_imgs = os.listdir(normal_dir)

  writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_file_path,"eval.tfrecords"))

  classify = os.listdir(data_file_path)
  # disgusting mac
  if ".DS_Store" in classify:
    classify.remove(".DS_Store")

  eval_image_count = 0
  for image_dir in classify:
    image_dir_path = os.path.join(data_file_path,image_dir)
    images = os.listdir(image_dir_path)
    if ".DS_Store" in images:
      images.remove(".DS_Store")
    for image in images:
      image_path = os.path.join(image_dir_path,image)
      try:
        img = Image.open(image_path)
        # w,h = img.size
        # img = img.crop([w//2-32,h//2-32,w//2+32,h//2+32])
        img = img.resize((image_width,image_height))
        img_raw = img.tobytes()
        example = tf.train.Example(
        features=tf.train.Features(feature={
          "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(image_dir)])),
          'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
        eval_image_count+=1
      except:
        print("error:",image_path)
  writer.close()
  return eval_image_count

def create_predict_record(tfrecord_file_path=config.tfrecord_file_path, data_file_path=config.data_file_path, image_width=config.image_width,image_height=config.image_height):
  print("create_predict_record--------------------start")
  data_file_path = os.path.join(data_file_path,"predict")
  if not os.path.exists(data_file_path):
    os.makedirs(data_file_path)
  imgs = os.listdir(data_file_path)

  if not os.path.exists(tfrecord_file_path):
    os.makedirs(tfrecord_file_path)

  writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_file_path,"predict.tfrecords"))

  predict_image_count = 0
  for image in imgs:
    image_path = os.path.join(data_file_path,image)
    try:
      img = Image.open(image_path)
      # w,h = img.size
      # img = img.crop([w//2-32,h//2-32,w//2+32,h//2+32])
      img = img.resize((image_width,image_height))
      img_raw = img.tobytes()
      example = tf.train.Example(
      features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
      }))
      writer.write(example.SerializeToString())
      predict_image_count += 1
    except:
      print("error:",image_path)

  
  writer.close()
  return predict_image_count,imgs

def create_predict_record_from_web(file_paths, tfrecord_file_path=config.tfrecord_file_path, image_width=config.image_width,image_height=config.image_height):
  predict_image_count=0
  if not os.path.exists(tfrecord_file_path):
    os.makedirs(tfrecord_file_path)
  writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_file_path,"predict.tfrecords"))
  for image_path in file_paths:
    try:
      img = Image.open(image_path)
      # w,h = img.size
      # img = img.crop([w//2-32,h//2-32,w//2+32,h//2+32])
      img = img.resize((image_width,image_height))
      img_raw = img.tobytes()
      example = tf.train.Example(
      features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
      }))
      predict_image_count += 1
      writer.write(example.SerializeToString())
    except:
      print("error:",image_path)
  writer.close()
  return predict_image_count


def parse_feature(value):
  feature = tf.parse_single_example(
    value,
    features={
      'label': tf.FixedLenFeature([], tf.int64),
      'image':tf.FixedLenFeature([], tf.string)
    }
  )
  feature['image'] = tf.decode_raw(feature['image'],tf.uint8)
  feature['image'] = tf.cast(feature['image'],tf.float32)
  feature['image'] = tf.reshape(feature['image'],[config.image_width,config.image_height,config.image_channel])
  feature['image'] = tf.image.random_flip_left_right(feature['image'])
  feature['image'] = tf.image.random_brightness(feature['image'], max_delta=63)
  feature['image'] = tf.image.random_contrast(feature['image'],lower=0.2, upper=1.8)
  feature['image'] = tf.image.per_image_standardization(feature['image'])
  feature['image'].set_shape([config.image_width,config.image_height,config.image_channel])
  
  label = feature.pop("label")
  return (feature,label)


def read_and_decode(flag, tfrecord_file_path=config.tfrecord_file_path, tfrecord_file_count=config.tfrecord_file_count):
  if flag == 'train':
    tfrecord_files = os.listdir(tfrecord_file_path)
    train_tfrecord_file_num = 0
    for i in tfrecord_files:
      if 'train' in i:
        train_tfrecord_file_num += 1
    files = []
    if train_tfrecord_file_num < tfrecord_file_count:
      for j in tfrecord_files:
        if 'train' in j:
          files.append(os.path.join(tfrecord_file_path,j))
    else:
      while len(files) < tfrecord_file_count:
        if os.path.join(tfrecord_file_path,'train'+str(random.randint(0,train_tfrecord_file_num-1))+'.tfrecords') not in files:
          files.append(os.path.join(tfrecord_file_path,'train'+str(random.randint(0,train_tfrecord_file_num-1))+'.tfrecords'))
  elif flag == 'eval':
    files = os.path.join(tfrecord_file_path,'eval.tfrecords')
  dataset = tf.data.TFRecordDataset(files)
  dataset = dataset.map(parse_feature)
  return dataset

def predict_parse_feature(value):
  feature = tf.parse_single_example(
    value,
    features={
      'image':tf.FixedLenFeature([], tf.string)
    }
  )
  feature['image'] = tf.decode_raw(feature['image'],tf.uint8)
  feature['image'] = tf.cast(feature['image'],tf.float32)
  feature['image'] = tf.reshape(feature['image'],[config.image_width,config.image_height,config.image_channel])
  feature['image'] = tf.image.per_image_standardization(feature['image'])
  feature['image'].set_shape([config.image_width,config.image_height,config.image_channel])
  
  return feature

def predict_read_and_decode(tfrecord_file_path=config.tfrecord_file_path):
  files = os.path.join(tfrecord_file_path,'predict.tfrecords')
  dataset = tf.data.TFRecordDataset(files)
  dataset = dataset.map(predict_parse_feature)
  return dataset