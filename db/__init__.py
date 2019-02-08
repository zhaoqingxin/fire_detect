import MySQLdb
import os

config_path = "config."+os.environ["PYTHON_ENV"]
import importlib
config = importlib.import_module(config_path)


class Phoenix():
  def __init__(self):
    self.db = MySQLdb.connect(host=config.db['host'], port=config.db['port'], user=config.db['user'], password=config.db['password'], db=config.db['schema'], charset='utf8' )
    self.cursor = self.db.cursor()
  def insert(self, flag, data):
    if flag=='train':
      sql = """INSERT INTO train (train_image_count, train_step, epoch, duration) VALUES (%s, %s, %s, %s)"""
      self.cursor.execute(sql,data)
      self.db.commit()
    elif flag=='evaluate':
      sql = """INSERT INTO evaluate (accuracy, loss, global_step) VALUES (%s, %s, %s)"""
      self.cursor.execute(sql,data)
      self.db.commit()
    elif flag=='predict':
      sql = """INSERT INTO predict (name, classify, probabilities) VALUES (%s, %s, %s)"""
      self.cursor.executemany(sql,data)
      self.db.commit()
  def close(self):
    self.db.close()


phoenix = Phoenix()
