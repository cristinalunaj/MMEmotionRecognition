import numpy
import torch
import torch.nn as nn
import sys

from sklearn.utils import compute_class_weight

from src.Video.models.sequenceLearning.environment import DEVICE


def get_class_weights(y):
	"""
		Compute normalized weigths for every class in dataset labels

	Args:
		y: (N,) list of labels

	Return:
		a dict with the label - keys and label_frequency - values
	"""
	classes = get_class_labels(y)
	weigths = compute_class_weight(class_weight="balanced", 
		classes=classes, y=y)
	d = {c: w for c, w in zip(numpy.unique(y), weigths)}
	return d


def get_class_labels(y):
	"""
		Unique labels within list of classes

	Args:
		y: (N,) list of labels

	Return:
		a (M,) no.ndarray of unique labels
	"""
	return numpy.unique(y)


def class_weights(targets, to_pytorch=False):
	"""
		Compute a set of labels according to the relative
		importance of every label in terms of frequency within the
		dataset under examination.

	Args:
		targets: (N,) list of dataset labels
		to_pytorch: wheteher to convert to torch.Tensor

	Return:
		a (N,) list or torch.Tensor of dataset labels
	"""
	w = get_class_weights(targets)
	labels = get_class_labels(targets)
	if to_pytorch:
		return torch.FloatTensor([w[l] for l in sorted(labels)])
	return labels


def get_criterion(task_name, weights=None):
	"""
		Get the appropiate training criterion given a task
		objective

	Args:
		task_name: str, task goal [regression|binary|multilabel]
		weights: (N,) torch.Tensor of dataset labels

	Return:
		a nn.Loss loss function criterion
	"""
	if task_name == "regression":
		return torch.nn.MSELoss()

	if task_name == "multilabel":
		return torch.nn.CrossEntropyLoss(weight=weights)

	if task_name == "binary":
		return torch.nn.BCEWithLogitsLoss()

	raise IOError("Undefined task - Missing criterion!")


def get_optimizer(parameters, lr, weight_decay):
	"""
		Initiate Adam optimizer with fixed parameters

	Args:
		parameters: filter, parameters to optimize
		lr: float, initial learning rate
		weight_decay: float, between 0.0 and 1.0

	Return:
		a torch.optim.Adam optimizer
	"""
	return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def get_pipeline(task_name, criterion, eval_phase=False):
	"""
		A wrapper to carry out a forward pass of a model in 
		several cases

	Args:
		task_name: str, task goal [regression|binary|multilabel]
		criterion: a nn.Loss loss function
		eval_phase: bool, whether is training phase

	Return:
		a callable to perform model forward pass
	"""
	def pipeline(model, batch):
		"""
			Compute forward pass variables within a model

		Args:
			model: a torch.nn.Module
			batch: (B, L, H) batch of samples to pass to model

		Return:
			a (B, out_size) torch.Tensor of outputs
			a (B, num_classes) torch.Tensor of ground truth labels
			a (B, L) torch.Tensor or attention values
			a int, loss function value on current batch
		"""
		inputs, labels, lengths, indices, _ = batch
		if task_name is "regression":
			labels = labels.float()
		inputs = inputs.to(DEVICE)
		labels = labels.to(DEVICE)
		lengths = lengths.to(DEVICE)

		outputs, attentions = model(inputs, lengths)

		if eval_phase:
			return outputs, labels, attentions, None

		if task_name == "binary":
			loss = criterion(outputs.view(-1), labels.float())
		else:
			loss = criterion(outputs.squeeze(), labels)

		return outputs, labels, attentions, loss
		
	return pipeline


def epoch_progress(loss, epoch, batch, batch_size, dataset_size):
	count = batch * batch_size
	bar_len = 40
	filled_len = int(round(bar_len * count / float(dataset_size)))
	bar = '=' * filled_len + '-' * (bar_len - filled_len)
	status = 'Epoch {}, Batch Loss ({}): {:.4f}'.format(
		epoch, batch, loss)
	_progress_str = "\r \r [{}] ... {}".format(bar, status)
	sys.stdout.write(_progress_str)
	sys.stdout.flush()


def unfreeze_module(module, optimizer):
	"""
		Turn into trainable a torch nn.Module
	"""
	for param in module.parameters():
		param.requires_grad = True


