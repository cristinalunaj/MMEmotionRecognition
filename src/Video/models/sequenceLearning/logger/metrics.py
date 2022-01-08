import math
import numpy

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score


def calc_pearson(y, y_hat):
	"""
		Compute Pearson correlation coefficient

	Args:
		y: (N,) list of ground_truth values
		y_hat: (N,) list of predicted values

	Return:
		a float, pearson correlation coefficient
	"""
	score, p_value = pearsonr(y, y_hat)
	if math.isnan(score):
		return 0
	else:
		return score


def calc_spearman(y, y_hat):
	"""
		Compute Spearman correlation coefficient

	Args:
		y: (N,) list of ground_truth values
		y_hat: (N,) list of predicted values

	Return:
		a float, spearman correlation coefficient
	"""
	score, p_value = spearmanr(y, y_hat)
	if math.isnan(score):
		return 0
	else:
		return score


def get_metrics(task):
	"""
		Wrapper to have at hand what metrics and how to monitor
		them at training time.

	Args:
		task: str, task goal [regression|binary|multilabel]

	Return:
		a tuple:
			- a metrics to eval on dict
			- a monitoring metric s name
			- a str, denoting whether to seek max or min value of
			the monitoring metric
	"""
	metrics_ = {
		"regression": {
			"pearson": calc_pearson,
			"spearman": calc_spearman
		},
		"binary": {
			"acc": lambda y, y_hat: accuracy_score(y, y_hat),
			"precision": lambda y, y_hat: precision_score(y, y_hat,
				average="weighted", zero_division=0),
			"recall": lambda y, y_hat: recall_score(y, y_hat, 
				average="weighted"),
			"f1": lambda y, y_hat: f1_score(y, y_hat, average="weighted")
		},
		"multilabel": {
			"acc": lambda y, y_hat: accuracy_score(y, y_hat),
			"precision": lambda y, y_hat: precision_score(y, y_hat,
				average="weighted", zero_division=0),
			"recall": lambda y, y_hat: recall_score(y, y_hat, 
				average="weighted"),
			"f1": lambda y, y_hat: f1_score(y, y_hat, average="weighted")
		}
	}

	monitor_ = {
		"regression": "spearman",
		"binary": "f1",
		"multilabel": "f1"
	}

	mode_ = {
		"regression": "max",
		"binary": "max",
		"multilabel": "max"
	}

	return metrics_[task], monitor_[task], mode_[task]
