import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es, rs = 0, 0
    if percent > 0:
      es = s / (percent)
      rs = es - s
    return '(%s - %s)' % (asMinutes(s), asMinutes(rs))

def timePassed(since):
    now = time.time()
    dt = now - since
    return f'{dt*1000:.2f}ms'

def tokenSpeed(since, proc):
    now = time.time()
    dt = now - since
    es = proc / dt
    return f'{es:.2f}'