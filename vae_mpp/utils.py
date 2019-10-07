from datetime import datetime

def print_log(*args):
    print("[{}]".format(datetime.now()), *args)
