import multiprocessing
workers = multiprocessing.cpu_count()
# workers = 1
bind = '0.0.0.0:24680'
max_requests = 10000
accesslog = '/Users/zhaoqingxin/hzmt/huazhengmingtian/logs/gunicorn_access.log'
errorlog = '/Users/zhaoqingxin/hzmt/huazhengmingtian/logs/gunicorn_error.log'
loglevel='error'