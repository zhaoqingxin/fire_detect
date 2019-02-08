# # train images store dir
data_file_path = "/Users/zhaoqingxin/hzmt/data"

# # images transfor to tfrecord file 
tfrecord_file_path = "/Users/zhaoqingxin/hzmt/tfrecords"
# train_tfrecord_file_path = "/Users/zhaoqingxin/Desktop/python-file/demo/tfrecords/train.tfrecords"
# eval_tfrecord_file_path = "/Users/zhaoqingxin/Desktop/python-file/demo/tfrecords/eval.tfrecords"
# predict_tfrecord_file_path = "/Users/zhaoqingxin/Desktop/python-file/demo/tfrecords/predict.tfrecords"

# # dir where machine-learn cnn model store, incloud model checkpoint summary and variable
model_dir="/Users/zhaoqingxin/hzmt/convnet_model"

db={
  'active':True,
  'host':'127.0.0.1',
  'port':3306,
  'user':'root',
  'password':'123456',
  'schema':'phoenix'
}

# # train images store dir
# data_file_path = "e:/hzmt/data"

# # images transfor to tfrecord file 
# tfrecord_file_path = "e:/hzmt/tfrecords"

# # dir where machine-learn cnn model store, incloud model checkpoint summary and variable
# model_dir="e:/hzmt/convnet_model"

# image count per tfrecord file
tfrecord_image_count = 2000

# tfrecord file number per train time
tfrecord_file_count = 4

# adjust image to unify size
image_height = 32
image_width = 32
image_channel = 3

# batch for compute
batch_size = 20

# train total step
epoch = 4


params = {
  'learn_rate':0.001,
  'classify':2,
  'stddev':0.1,
  'c_layers':[
    {
      'filters':32,
      'kernel_size':[5, 5],
      'pool_size':[2, 2],
      'strides':2
    },
    # {
    #   'filters':128,
    #   'kernel_size':[5, 5],
    #   'pool_size':[2, 2],
    #   'strides':2
    # },
    {
      'filters':64,
      'kernel_size':[5, 5],
      'pool_size':[2, 2],
      'strides':2
    }
  ],
  'd_layers':[
    {
        'units':512,
        'dropout':0.6
    }
  ]
}
