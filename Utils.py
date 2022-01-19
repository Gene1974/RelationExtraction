import sys
import time

def logger(content):
    time_stamp = time.strftime("%m-%d %H:%M:%S", time.localtime())
    sys.stdout.write('[{}] {}\n'.format(time_stamp, content))
    sys.stderr.write('[{}] {}\n'.format(time_stamp, content))
