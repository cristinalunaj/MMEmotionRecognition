import json
import math
import os
import pandas
import pickle
import time
import torch

from src.Video.models.sequenceLearning.utils.prediction import compute_confusion_matrix

from src.Video.models.sequenceLearning.environment import TRAINED_PATH, EXPERIMENT_PATH


def log_training(trainer, name, description, label_transformer=None):
	"""
		Display general train-eval information on screen

	Args:
		trainer: a Trainer object
		name: str, name to associate results to
		description: str, name description
	"""
	results = {}
	scores = {k: v for k, v in trainer.scores.items() if k != "loss"}

	results["name"] = name
	results["description"] = description
	results["scores"] = scores

	# TODO: Solve cache issues! If current pipeline throws
	# any problem, remove cache and out files!
	path = os.path.join(EXPERIMENT_PATH, trainer.config["name"])
	if not os.path.isdir(EXPERIMENT_PATH):
		os.makedirs(EXPERIMENT_PATH)
	json_file = path + ".json"
	try:
		with open(json_file) as f:
			data = json.load(f)
	except:
		data = []
	data.append(results)
	with open(json_file, 'w') as f:
		json.dump(data, f)

	results_ = []
	for result in data:
		res_ = {k: v for k, v in result.items() if k != "scores"}
		for score_name, score in result["scores"].items():
			for tag, values in score.items():
				res_["_".join([score_name, tag, "min"])] = min(values)
				res_["_".join([score_name, tag, "max"])] = max(values)
		results_.append(res_)
	with open(path + ".csv", "w") as f:
		pandas.DataFrame(results_).to_csv(
			f, sep=',', encoding='utf-8')

	splits, confusion_matrix = compute_confusion_matrix(trainer)
	for split, cm in zip(splits, confusion_matrix):
		cm.to_csv('-'.join(
			[path, name, description, split, 'confusion_matrix.csv']))




def log_fold_training(trainer):
	"""
		Summarizes information relative to fold training.

	Args:
		trainer: a Trainer object

	Return:
		a dict of metrics in a fold
	"""
	parts_fold = [set(v.keys()) for v in trainer.scores.values()]
	partitions = set()
	[partitions.update(x) for x in parts_fold]
	
	if len(partitions) == 1:
		try:
			scores = {k: v["train"] for k, v in trainer.scores.items()}
		except KeyError:
			scores = {k: v["test"] for k, v in trainer.scores.items()}
	else:
		scores = {k: v["val"] for k, v in trainer.scores.items()}

	fold_result = {}
	if trainer.task != "regression":
		p_max_f1 = scores["f1"].index(max(scores["f1"]))
		fold_result["acc"] = scores["acc"][p_max_f1]
		fold_result["precision"] = scores["precision"][p_max_f1]
		fold_result["recall"] = scores["recall"][p_max_f1]
		fold_result["f1"] = scores["f1"][p_max_f1]
		fold_result["loss"] = scores["loss"][p_max_f1]
	else:
		p_max_pearson = scores['pearson'].index(max(scores['pearson']))
		fold_result['pearson'] = scores['pearson'][p_max_pearson]
		fold_result['spearman'] = scores['spearman'][p_max_pearson]
		fold_result['loss'] = scores['loss'][p_max_pearson]
	return fold_result


def log_K_folds_training(trainer, name, description, results):
	"""
		Display on terminal overall information about learning
		process

	Args:
		trainer: a Trainer object
		name: str, name to associate results to
		description: str, name description
		results: list of KFolds results
	"""
	print('\n  MODEL: {}'.format(name))
	print('  CONFIG: {}'.format(description))
	print('  FINAL CROSS VALIDATION RESULTS')
	n_folds = len(results)

	if trainer.task != "regression":
		print('{:<20}'.format('<fold>') +
			'{:<20}'.format('<acc>') +
			'{:<20}'.format('<precision>') +
			'{:<20}'.format('<recall>') +
			'{:<20}'.format('<f1>') +
			'{:<20}'.format('<loss>'))

		valid_fold = -1
		for fold in range(n_folds):
			if results[fold] is not None:
				print('{:<20}'.format(str(fold+1)) +
					'{:<20.5f}'.format(results[fold]["acc"]) +
					'{:<20.5f}'.format(results[fold]["precision"]) +
					'{:<20.5f}'.format(results[fold]["recall"]) +
					'{:<20.5f}'.format(results[fold]["f1"]) +
					'{:<20.5f}'.format(results[fold]["loss"]))
				valid_fold = fold

		avg_results = {}
		for key in results[valid_fold]:
			avg_results[key] = sum(d[key] for d in results if \
				d is not None) / sum(x is not None for x in results)

		print('{:<20}'.format('AVG') +
			'{:<20.5f}'.format(avg_results["acc"]) +
			'{:<20.5f}'.format(avg_results["precision"]) +
			'{:<20.5f}'.format(avg_results["recall"]) +
			'{:<20.5f}'.format(avg_results["f1"]) +
			'{:<20.5f}'.format(avg_results["loss"]))
		
	else:
		print('{:<20}'.format('<fold>') +
			'{:<20}'.format('<pearson>') +
			'{:<20}'.format('<spearman>') +
			'{:<20}'.format('<loss>')
			)

		valid_fold = -1
		for fold in range(n_folds):
			if results[fold] is not None:
				print('{:<20}'.format(str(fold + 1)) +
					'{:<20.5f}'.format(results[fold]["pearson"]) +
					'{:<20.5f}'.format(results[fold]["spearman"]) +
					'{:<20.5f}'.format(results[fold]["loss"])
					)
				valid_fold = fold

		avg_results = {}
		for key in results[valid_fold]:
			avg_results[key] = sum(d[key] for d in results if \
				d is not None) / sum(x is not None for x in results)
		print('{:<20}'.format('AVERAGE') + 
			'{:<20.5f}'.format(avg_results['pearson']) +
			'{:<20.5f}'.format(avg_results['spearman']) +
			'{:<20.5f}'.format(avg_results['loss'])
			)
		
		os.makedirs(EXPERIMENT_PATH, exist_ok=True)
		path = os.path.join(EXPERIMENT_PATH, name)
		json_file = path + description + '.json'

		data = []
		data.append(results)
		data.append(avg_results)
		with open(json_file, 'w') as f:
			json.dump(data, f)


class MetricWatcher:
	"""
		Base class that monitors a given metric and assess improvement

	Args:
		metric: dict, metrics to keep track of
		monitor: str, name of the monitoring metric
		mode: str, whether seek max or min in metric
		base: float, baseline to keep models up from
	"""
	def __init__(self, metric, monitor, mode, base=None, min_change=0.0):
		self.best = base
		self.metric = metric
		self.mode = mode
		self.monitor = monitor
		self.min_change = min_change

		self.scores = None

	def has_improved(self):
		"""
			Assess monitor_metric improvement
		"""
		last_metric_val = self.scores[self.metric][self.monitor][-1]
		if not self.best or math.isnan(self.best):
			self.best = last_metric_val
			return True
		if self.mode == "min" and last_metric_val < (self.best - self.min_change):
			self.best = last_metric_val
			return True
		if self.mode == "max" and last_metric_val > (self.best + self.min_change):
			self.best = last_metric_val
			return True

		return False


class Checkpoint(MetricWatcher):
	"""
		Logger relative to keep track of everything that happens
		while training a model (metrics, improvements, records...)

	Args:
		name: str, model s name
		model: nn.Module, model under training
		monitor: str, name of the monitoring metric
		metric: dict, metrics to keep track of
		model_conf:
		mode: str, whether seek max or min in metric
		model_dir: str, output directory to save model
		base: float, baseline to keep models up from
		timestamp: bool, whether to add timestamp to checkpoint name
		scorestamp: bool, whether to add scorestamp to checkpoint name
		keep_best: bool, whether to keep all or just best model version
	"""
	def __init__(self, name, model, monitor, metric, model_conf, mode,
		model_dir=None, base=None, timestamp=None, scorestamp=None, 
		keep_best=True, labelTransformer = None):
		MetricWatcher.__init__(self, metric, monitor, mode, base)

		self.name = name
		self.model_dir = model_dir
		self.model = model
		self.model_conf = model_conf
		self.timestamp = timestamp
		self.scorestamp = scorestamp
		self.keep_best = keep_best
		self.last_saved = None
		self.labelTransformer = labelTransformer

		if not self.model_dir:
			self.model_dir = TRAINED_PATH
		print("SAVING MODEL IN ", self.model_dir)

	def _define_cp_name(self):
		"""
			Set the checkpoints names
		"""
		filename = self.name
		if self.scorestamp:
			filename += "-{:.4f}".format(self.best)
		if self.timestamp:
			filename += time.strftime("-%Y_%m_%d_%H_%M")
		return filename

	def _save_cp(self):
		"""
			Saving model functionality. A typical checkpoint saves
				- the model weights and architecture
				- model s config to further retraining
		"""
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir, exist_ok=True)

		name = self._define_cp_name()
		file_cp = os.path.join(self.model_dir, name + ".model")
		file_conf = os.path.join(self.model_dir, name + ".conf")
		#file_labels = os.path.join(self.model_dir, name + "_labels.pkl")

		if self.keep_best and self.last_saved is not None:
			os.remove(self.last_saved["model"])
			os.remove(self.last_saved["config"])
			#os.remove(self.last_saved["labelTransformer"])

		self.last_saved = {
			"model": file_cp,
			"config": file_conf,
			#"labelTransformer": file_labels
		}

		torch.save(self.model, file_cp)
		with open(file_conf, 'wb') as file:
			pickle.dump(self.model_conf, file)

		# if(self.labelTransformer!=None):
		# 	self.labelTransformer.save_labelTransformer(file_labels)

	def assess(self):
		"""
			Assess whether the model has improved upon selected metric.
			If True, save checkpoint. Dismiss otherwise.
		"""
		if self.has_improved():
			print("Improved model ({}:{:.4f}) -- Saving checkpoint".format(
				self.metric, self.best))
			self._save_cp()


class EarlyStop(MetricWatcher):
	"""
		Early stopping procedure. Stops training after N training epochs
		have not achieved substantial improvement upon monitor metric

	Args:
		metric: dict, metrics to keep track of
		monitor: str, name of the monitoring metric
		mode: str, whether seek max or min in metric
		patience: int, early stopping criterion
		min_change: TODO float, minimum improvement to consider it an actual improvement
	"""
	def __init__(self, metric, monitor, mode, patience=0, min_change=0.0):
		MetricWatcher.__init__(self, metric, monitor, mode, min_change=min_change)
		self.patience = patience
		self.left = patience
		self.best = None

	def stop(self):
		"""
			Whether we stop training
		"""
		if self.has_improved():
			self.left = self.patience
		else:
			self.left -= 1

		print("  Patience left: {}, best: {:0.4f}".format(
			self.left, self.best))
		return self.left < 0

