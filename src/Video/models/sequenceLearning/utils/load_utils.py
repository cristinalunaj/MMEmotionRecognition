import numpy
import pickle
import torch

from os.path import exists, join, split, splitext

from src.Video.models.sequenceLearning.environment import EMBEDDINGS_PATH
from src.Video.models.sequenceLearning.environment import TRAINED_PATH


def get_pretrained(pretrained, trained_path=""):
	if isinstance(pretrained, list):
		pretrained_models = []
		pretrained_config = []
		for pt in pretrained:
			pt_model, pt_conf = load_pretrained_model(pt, trained_path)
			pretrained_models.append(pt_model)
			pretrained_config.append(pt_conf)
		return pretrained_models, pretrained_config

	return load_pretrained_model(pretrained, trained_path)


def load_pretrained_model(name, trained_path=""):
	if(trained_path==""):
		trained_path = TRAINED_PATH
	model_path = join(trained_path, "{}.model".format(name))
	conf_path = join(trained_path, "{}.conf".format(name))

	try:
		model = torch.load(model_path)
	except:
		model = torch.load(model_path, map_location=torch.device('cpu'))

	model_conf = pickle.load(open(conf_path, 'rb'))
	return model, model_conf


def load_embeddings(config):
	word_vectors = join(EMBEDDINGS_PATH, "{}.txt".format(
		config["embeddings_file"]))
	word_vector_size = config["input_size"]
	print("  Loading word embeddings from vocabulary file...")
	disable_cache = config.get("disable_cache", False)
	return load_word_vectors(word_vectors,
		word_vector_size, disable_cache)


def file_cache_name(file):
	"""
		Get cache file name from file basename
	"""
	head, tail = split(file)
	filename, ext = splitext(tail)
	return join(head, filename + ".p")


def load_cache_word_vectors(file):
	with open(file_cache_name(file), 'rb') as file:
		return pickle.load(file)


def write_cache_word_vectors(file, data):
	"""
		Write out a cache file optimized for later readings.

	Args:
		file: str, file name
		data: tuple of (word2idx, idx2word, embeddings)
	"""
	with open(file_cache_name(file), 'wb') as f:
		pickle.dump(data, f)


def load_word_vectors(file, dim, disable_cache=True):
	"""
		Read in word vectors from a vocabulary text file

	Args:
		file: str, name of the vocabulary file
		dim: int, embedding size
		disable_cache: bool, keep a cache copy of the data

	Return:
		word2idx: dict, mapping word to index
		idx2word: dict, mapping index to word
		embeddings: numpy.ndarray, word embeddings matrix
	"""
	try:
		cache = load_cache_word_vectors(file)
		print('  Successfully loaded word embeddings from cache')
		return cache
	except OSError:
		print("  Did not find embeddings cache file {}".format(file))

	if exists(file):
		print("  Indexing file {}...".format(file))

		word2idx = {}
		idx2word = {}
		embeddings = []

		# We reserve first embeddings as a zero-padding
		# embedding with idx=0
		embeddings.append(numpy.zeros(dim))
		# Does the embeddings file have header?
		header = False

		with open(file, "r", encoding="utf-8") as f:
			for i, line in enumerate(f):
				if i == 0:
					if len(line.split()) < dim:
						header = True
						continue
				values = line.split(" ")
				word = values[0]
				try:
					vector = numpy.asarray(values[1:], dtype='float32')
				except ValueError:
					vector = numpy.array([float(x) for x in values[1:-1]])
					assert len(vector) == dim

				index = i - 1 if header else i

				idx2word[index] = word
				word2idx[word] = index
				embeddings.append(vector)

			# Add an <unk> token for OOV words
			if "<unk>" not in word2idx:
				idx2word[len(idx2word) + 1] = "<unk>"
				word2idx["<unk>"] = len(word2idx) + 1
				embeddings.append(
					numpy.random.uniform(low=-0.05, high=0.05, size=dim))

			print('Embeddings sizes found: ', set([
				len(x) for x in embeddings]))
			print("Found {} word vectors.".format(len(embeddings)))
			embeddings = numpy.array(embeddings, dtype='float32')

		if not disable_cache:
			write_cache_word_vectors(file, (word2idx, idx2word, embeddings))
		return word2idx, idx2word, embeddings

	else:
		raise OSError("{} not found!".format(file))

