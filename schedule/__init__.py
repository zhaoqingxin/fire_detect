import time

def tesk():
    print("schedule---------------------------------------run")

def schedule(date):
  seconds = 10
  while(True):
    time.sleep(seconds)
    tesk()

