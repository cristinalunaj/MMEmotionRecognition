import argparse, os
import torch
#REDUCE RANDOMNESS:
import random
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_torch(seed=2020):
    seed_libs(2020)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def seed_tf(seed=2020):
    # tf 1...
    # from tensorflow import set_random_seed
    # set_random_seed(seed)
    #tf 2...
    import tensorflow as tf
    tf.random.set_seed(seed)
    seed_libs(seed=seed)

def seed_libs(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
