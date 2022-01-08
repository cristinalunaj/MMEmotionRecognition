import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device {}".format(str(DEVICE).upper()))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TRAINED_PATH = os.path.join(BASE_PATH, "out/trained")
EXPERIMENT_PATH = os.path.join(BASE_PATH, "out/experiments")
DATA_DIR = os.path.join(BASE_PATH, "datasets")
EMBEDDINGS_PATH = os.path.join(BASE_PATH, "embeddings")
CACHE_PATH = os.path.join(BASE_PATH, "_cache")