import json
import math
import os
import pandas
import pickle
import time
import torch

from src.Video.models.sequenceLearning.utils.prediction import compute_confusion_matrix

from src.Video.models.sequenceLearning.environment import TRAINED_PATH, EXPERIMENT_PATH


def log_inference(tester, name, description="", out_path=""):
	"""
		Saves on disk inference results.

	Args:
		tester: a Tester object
		name: str, name to associate results to
		description: str, name description
	"""
	if(out_path==""):
		out_path = EXPERIMENT_PATH

	for dataset, output in tester.preds.items():
		results = pandas.DataFrame.from_dict(output)
		path = os.path.join(
			out_path, tester.config["name"] + '-' + dataset + description)
		with open(path + ".csv", "w") as f:
			results.to_csv(f, sep="\t", encoding='utf-8', 
				float_format='%.3f', index=False)

		with open(path + "-predictions.csv", "w") as f:
			results[["tag", "y_hat"]].to_csv(
				f, index=False, float_format='%.3f', header=False)

		with open(path + "-posteriors.csv", "w") as f:
			results[["tag", "posteriors"]].to_csv(
				f, index=False, float_format='%.3f', header=False)

		with open(path + "-featAttention.csv", "w") as f:
			results[["tag", "attentions"]].to_csv(
				f, index=False, float_format='%.3f', header=False)


def log_evaluation(tester, name, description):
	"""
		Saves on disk evaluation results.

	Args:
		testet: a Tester object
		name: str, name to associate results to
		description: str, name description
	"""
	for dataset, output in tester.preds.items():
		results = pandas.DataFrame.from_dict(output)
		path = os.path.join(
			EXPERIMENT_PATH, tester.config["name"] + '-' + dataset)
		with open(path + ".csv", "w") as f:
			results.to_csv(f, sep="\t", encoding='utf-8',
				float_format='%.3f', index=False)

	splits, confusion_matrix = compute_confusion_matrix(tester)
	for split, cm in zip(splits, confusion_matrix):
		cm.to_csv('-'.join(
			[path, name, description, split, 'confusion_matrix.csv']))