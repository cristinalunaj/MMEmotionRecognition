import torch

from torch.nn import ModuleList
from torch.nn.utils import clip_grad_norm_

from src.Video.models.sequenceLearning.environment import TRAINED_PATH
from src.Video.models.sequenceLearning.utils.training import epoch_progress
from src.Video.models.sequenceLearning.utils.training import unfreeze_module
from src.Video.models.sequenceLearning.utils.prediction import predict


class Trainer:
	"""
		Wrapper for typical training pipelines of a DL model.
		Basically an abstraction for the whole training process.

	Args:
		model: a nn.Module torch model
		loaders: dict of torch dataloaders by phase
		optimizer:
		pipeline: a callback, output computing wrapper
		config: dict, parameters of training process
		task: str, task goal [regression|binary|multilabel]
		metrics: dict of metrics to observe
		eval_train: bool, whether to eval on validation after every epoch
		checkpoint: 
		early_stopping: EarlyStop instance
	"""
	def __init__(self, model, loaders, optimizer, pipeline, config,
		task, metrics, eval_train=True, checkpoint=None, early_stopping=None):
		self.model = model
		self.loaders = loaders
		self.optimizer = optimizer
		self.pipeline = pipeline
		self.task = task
		self.config = config
		self.metrics = {} if not metrics else metrics
		self.eval_train = eval_train
		self.checkpoint = checkpoint
		self.early_stopping = early_stopping

		self.running_loss = 0.0
		self.epoch = 0

		dataset_names = list(self.loaders.keys())
		metric_names = list(self.metrics.keys()) + ['loss']
		self.scores = {
			m: {d: [] for d in dataset_names} for m in metric_names}

		if self.checkpoint is not None:
			self.checkpoint.scores = self.scores
		if self.early_stopping is not None:
			self.early_stopping.scores = self.scores

		self.preds = {d: {} for d in dataset_names}


	def train(self, epochs, unfreeze=-1):
		"""
			Main function within Trainer object. 
			This carries out the most of the computation
			phases in a typical DL stage.

		Args:
			epochs: int, max number of epochs to train on
			unfreeze: Layer to fine-tune (top-bottom)
		
		TODO: Refine the unfreezing mechanism
		"""
		print("[TRAINING]")
		if unfreeze == 0:
			print("  Unfreezing modules...")
			subnetwork = self.model.feature_extractor

			if isinstance(subnetwork, ModuleList):
				for fe in subnetwork:
					unfreeze_module(fe.encoder, self.optimizer)
					unfreeze_module(fe.attention, self.optimizer)
			else:
				unfreeze_module(subnetwork.encoder, self.optimizer)
				unfreeze_module(subnetwork.attention, self.optimizer)

		for epoch in range(epochs):
			self.epoch += 1
			self.train_epoch(self.loaders["train"])
			self.eval()

			if unfreeze > 0:
				if epoch == unfreeze:
					print("Unfreeze transfer-learning model...")
					subnetwork = self.model.feature_extractor
					if isinstance(subnetwork, ModuleList):
						for fe in subnetwork:
							unfreeze_module(
								fe.encoder, self.optimizer)
							unfreeze_module(
								fe.attention, self.optimizer)
						else:
							unfreeze_module(
								subnetwork.encoder, self.optimizer)
							unfreeze_module(
								subnetwork.attention, self.optimizer)

			self.checkpoint.assess()
			if self.early_stopping:
				if self.early_stopping.stop():
					print("[END OF TRAINING -- EARLY STOPPING...]")
					break

	def train_epoch(self, loader):
		"""
			Run a pass over an epoch of training data

		Args:
			loader: a torch.DataLoader 

		Return:
			Loss computed on the epoch
		"""
		self.model.train()
		running_loss = 0.0
		for i_batch, sample_batched in enumerate(loader, 1):
			self.optimizer.zero_grad()
			outputs, labels, attentions, loss = self.pipeline(
				self.model, sample_batched)
			loss.backward()
			# Gradient explosion preventing - norm clipping
			if len([m for m in self.model.modules() if \
				hasattr(m, "bidirectional")]) > 0:
				clip_grad_norm_(self.model.parameters(),
					self.config["clip_norm"])
			self.optimizer.step()
			running_loss += loss.item()

			epoch_progress(loss=loss.item(),
				epoch=self.epoch,
				batch=i_batch,
				batch_size=loader.batch_size,
				dataset_size=len(loader.dataset))

		#print()
		return running_loss

	def eval(self):
		"""
			Main evaluation function. Perform a metrics
			assessment over every dataset available to update
			metrics track recordings.
		"""
		for partition, loader in self.loaders.items():
			avg_loss, (y, y_hat), posteriors, atts, tags = self.eval_loader(
				loader)
			scores = self.__calc_scores(y, y_hat)
			self.__log_scores(scores, avg_loss, partition)
			scores['loss'] = avg_loss

			for name, value in scores.items():
				self.scores[name][partition].append(value)

			self.preds[partition] = {
				'tag': tags,
				'y': y,
				'y_hat': y_hat
			}

	def inference(self):
		"""
			Main inference function. Given data loaders,
			output the main attributes from the model.
		"""
		for partition, loader in self.loaders.items():
			avg_loss, (y, y_hat), post, attentions, tags = self.eval_loader(
				loader)
			self.preds[partition] = {
				'tag': tags,
				'y': y,
				'y_hat': y_hat,
				'posteriors': post,
				'attentions': attentions
			}

	def eval_loader(self, loader):
		"""
			Evaluate over a specific dataloader

		Args:
			loader: torch.DataLoader instance
		"""
		return predict(model=self.model,
			pipeline=self.pipeline,
			dataloader=loader,
			task=self.task,
			mode="eval")

	def __calc_scores(self, y, y_hat):
		return {name: metric(y, y_hat) for \
			name, metric in self.metrics.items()}

	def __log_scores(self, scores, loss, tag):
		"""
			Display metrics on console

		Args:
			scores: dict of {metric_name: value}
			loss: float, epoch average loss over epoch samples
			tag: str, dataloader s name
		"""
		print("\t{:6s} - ".format(tag), end=" ")
		for name, value in scores.items():
			print(name, '{:.4f}'.format(value), end=", ")
		print(" Loss: {:.4f}".format(loss))



