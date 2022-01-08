import os
import numpy
import pickle
import pandas as pd

from collections import Counter
from os.path import exists, join
from torch.utils.data import Dataset

from src.Video.models.sequenceLearning.environment import CACHE_PATH
from src.Video.models.sequenceLearning.utils.preprocessors import vectorize
from src.Video.models.sequenceLearning.utils.preprocessors import vectorize_pad


class BaseDataset(Dataset):
	""" Base class which extends on Pytorch's Dataset, to avoid
		boilerplate code and augment functionality with caching.

		Datasets based on this class must implement two basic functions:
			- __len__(self): Needed for batching, shuffling...
			- __getitem__(self, index): Get sample from position within
			dataset

	Args:
		X: list of samples
		labels: list of labels
		name: str, dataset name, required in caching. If None, disabled.
		preprocess:
		verbose: bool, whether to print out information
	"""
	def __init__(self, X, labels, name, preprocess=None, 
		max_length=0, verbose=True):
		self.data = X
		self.labels = labels
		self.name = name

		if preprocess:
			self.preprocess = preprocess

		self.data = self.load_preprocessed_data()
		self.set_max_length(max_length)

		if verbose:
			self.dataset_stats()

	def set_max_length(self, max_length):
		"""
			Set max length to largest sequence's length.
		"""
		self.max_length = max_length
		if max_length == 0:
			self.max_length = max([len(x) for x in self.data])

	def dataset_stats(self):
		raise NotImplementedError

	def preprocess(self, name, X):
		"""
			Basic preprocessing

		Args:
			X: list, sample data

		Return:
			a list of processed samples
		"""
		raise NotImplementedError

	def _get_cache_filename(self):
		return join(CACHE_PATH, "preprocessed_{}.p".format(
			self.name))

	def _write_cache(self, data):
		self._check_cache()
		cache_file = self._get_cache_filename()
		with open(cache_file, 'wb') as pk_file:
			pickle.dump(data, pk_file)

	def load_preprocessed_data(self):
		"""
			Quickly retrieve data from previously computed
			experiments and cache file.
		
		Return:
			a list of preprocessed samples
		"""
		if not self.name:	# no cache
			return self.preprocess(self.name, self.data)

		cache_file = self._get_cache_filename()
		if exists(cache_file):
			print("  Loading preprocessed data {} from cache!".format(
				self.name))
			with open(cache_file, 'rb') as f:
				return pickle.load(f)
		print("No cache file for dataset {}".format(self.name))
		data = self.preprocess(self.name, self.data)
		self._write_cache(data)
		return data

	@staticmethod
	def _check_cache():
		if not exists(CACHE_PATH):
			os.makedirs(CACHE_PATH, exist_ok=True)


class TokenDataset(BaseDataset):
	"""
		Dataset aimed at providing support for data types in 
		which we want to link a token-type element to a vector.
		Such tokens may include words, timestamps or other
		entities in which a lookup table token-vector is sought.

	Args:
		X: list of samples
		labels: list of labels
		name: str, dataset name, required in caching. If None, disabled.
		samples: list of sample names useful in inference
		word2idx: dict, match words and vocab embedding index
		max_length:	int, max embedding size
		preprocess: callable, preprocessing operation
		label_transformer: LabelTransformer
		unk_policy: str, value to assign to <unk> token embedding
		spell_corrector: callable, text processor
		verbose: bool, whether to display stats on screen
	"""
	def __init__(self, X, labels, name, samples, word2idx, max_length=0,
		preprocess=None, label_transformer=None, 
		unk_policy="random", spell_corrector=None, verbose=True):
		self.word2idx = word2idx
		super(TokenDataset, self).__init__(X, labels, name,
			preprocess, max_length, verbose)
		self.samples = samples
		self.label_transformer = label_transformer
		self.unk_policy = unk_policy
		self.spell_corrector = spell_corrector

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		"""
			Pytorch-friendly index-based sample picking

		Return:
			sample: ndarray, vector encoding
			label: str or float, class / score
			length: int, length of the dataitem sequence
			index: index dataitem, useful for visualization
		"""
		sample, label = self.data[index], self.labels[index]
		sample = vectorize(sequence=sample, 
			token2idx=self.word2idx,
			max_length=self.max_length,
			unk_policy=self.unk_policy,
			spell_corrector=self.spell_corrector)
		length = len(self.data[index])

		if self.label_transformer:
			label = self.label_transformer.transform(label)
		if isinstance(label, (list, tuple)):
			label = numpy.array(label)

		return sample, label, length, index, self.samples[index]

	def dataset_stats(self):
		tokens = Counter()
		for x in self.data:
			tokens.update(x)
		unks = {token: v for token, v in tokens.items() if \
			token not in self.word2idx}

		total_tokens = sum(tokens.values())
		total_unks = sum(unks.values())
		print("    Total Tokens: {}, total <unk>: {} ({:.2f}%)".format(
			total_tokens, total_unks, total_unks * 100 / total_tokens))
		print("    Unique tokens: {}, unique <unk>: {} {:.2f}%)".format(
			len(tokens), len(unks), len(unks) * 100 / len(tokens)))

		print("    Label statistics:")
		if isinstance(self.labels[0], float):
			print("      Mean = {:.3f}, Std = {:.3f}".format(
				numpy.mean(self.labels), numpy.std(self.labels)))
		elif isinstance(self.labels[0], str):
			counts = Counter(self.labels)
			stats = {k: "{:.2f}%".format(v * 100 / len(self.labels))
				for k, v in sorted(counts.items())}
			print('   ', stats)
		else:
			raise IOError("Label type not recognized!")


class PathDataset(BaseDataset):
	"""
		Dataset aimed at providing support for dataset
		in which the sample must be loaded from an external file,
		and thus rather than sequences present a path.

	Args:
		X: list of samples
		labels: list of labels
		name: str, dataset name, required in caching. If None, disabled.
		samples: list of sample names useful in inference
		max_length:	int, max embedding size
		preprocess: callable, preprocessing operation
		label_transformer: LabelTransformer
		verbose: bool, whether to display stats on screen
	"""
	def __init__(self, X, labels, name, samples, max_length=0, 
		preprocess=None, label_transformer=None, verbose=True):
		super(PathDataset, self).__init__(X, labels, name, preprocess,
			max_length, verbose)
		self.samples = samples
		self.label_transformer = label_transformer
		self.set_max_length(max_length)

	def set_max_length(self, max_length):
		"""
			Set max length to largest sequence's length.
		"""
		self.max_length = max_length
		if max_length == 0:
			self.max_length = max([len(numpy.load(x, allow_pickle=True))
				for x in self.data])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		"""
			Pytorch-friendly interface

		Return:
			sample: ndarray, vector sequence
			label: str or float, class / score
			length: int, length of the dataitem sequence
			index: indec dataitem, useful for visualization
		"""
		path, label = self.data[index], self.labels[index]
		sample = numpy.load(path, allow_pickle=True)
		length = len(sample)
		sample = vectorize_pad(sequence=sample,
			max_length=self.max_length)
		
		if self.label_transformer:
			label = self.label_transformer.transform(label)
		if isinstance(label, (list, tuple)):
			label = numpy.array(label)

		return sample, label, length, index, self.samples[index]

	def dataset_stats(self):
		assert len(self.data) == len(self.labels)
		total_samples = len(self.data)
		print("    Total Samples: {}".format(total_samples))

		print("    Label statistics:")
		if isinstance(self.labels[0], float):
			print("      Mean = {:.3f}, Std = {:.3f}".format(
				numpy.mean(self.labels), numpy.std(self.labels)))
		elif isinstance(self.labels[0], str):
			counts = Counter(self.labels)
			stats = {k: "{:.2f}%".format(v * 100 / len(self.labels))
				for k, v in sorted(counts.items())}
			print('   ', stats)
		else:
			raise IOError("Label type not recognized!")


class PathDatasetCSV(BaseDataset):
	"""
		Dataset aimed at providing support for dataset
		in which the sample must be loaded from an external file,
		and thus rather than sequences present a path.

	Args:
		X: list of samples
		labels: list of labels
		name: str, dataset name, required in caching. If None, disabled.
		samples: list of sample names useful in inference
		max_length:	int, max embedding size
		preprocess: callable, preprocessing operation
		label_transformer: LabelTransformer
		verbose: bool, whether to display stats on screen
	"""

	def __init__(self, X, labels, name, samples, max_length=0,
				 preprocess=None, label_transformer=None, verbose=True, sep=";"):

		self.sep = sep
		super(PathDatasetCSV, self).__init__(X, labels, name, preprocess,
										  max_length, verbose)
		self.samples = samples
		self.label_transformer = label_transformer



	def set_max_length(self, max_length):
		"""
			Set max length to largest sequence's length.
		"""
		self.max_length = max_length
		if max_length == 0:
			self.max_length = max([len(pd.read_csv(x, sep=self.sep, header=0))
								   for x in self.data])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		"""
			Pytorch-friendly interface

		Return:
			sample: ndarray, vector sequence
			label: str or float, class / score
			length: int, length of the dataitem sequence
			index: indec dataitem, useful for visualization
		"""
		path, label = self.data[index], self.labels[index]
		sample = pd.read_csv(path, sep=self.sep, header=0).values
		length = len(sample)
		sample = vectorize_pad(sequence=sample,
							   max_length=self.max_length)

		if self.label_transformer:
			label = self.label_transformer.transform(label)
		if isinstance(label, (list, tuple)):
			label = numpy.array(label)

		return sample, label, length, index, self.samples[index]

	def dataset_stats(self):
		assert len(self.data) == len(self.labels)
		total_samples = len(self.data)
		print("    Total Samples: {}".format(total_samples))

		print("    Label statistics:")
		if isinstance(self.labels[0], float):
			print("      Mean = {:.3f}, Std = {:.3f}".format(
				numpy.mean(self.labels), numpy.std(self.labels)))
		elif isinstance(self.labels[0], str):
			counts = Counter(self.labels)
			stats = {k: "{:.2f}%".format(v * 100 / len(self.labels))
					 for k, v in sorted(counts.items())}
			print('   ', stats)
		else:
			raise IOError("Label type not recognized!")