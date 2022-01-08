import nltk
import numpy

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


def vectorize(sequence, token2idx, max_length, unk_policy="random", 
	spell_corrector=None):
	"""
		Convert an array of tokens into an array of indices,
		with fixed length = max_length and right-sided zero padding

	Args:
		sequence: list of elements
		token2idx: dict, mapping between token and index
		max_length: max sequence length allowed
		unk_policy: str, how to handle OOV tokens
		spell_corrector: if unk_policy is "correct", then pass a
			callable to correct mispellings

	Return:
		list of indices with zero padding on the right side
	"""
	tokens = numpy.zeros(max_length).astype(int)
	sequence = sequence[:max_length]
	for i, token in enumerate(sequence):
		if token in token2idx:
			tokens[i] = token2idx[token]
		else:
			if unk_policy == "random":
				tokens[i] = token2idx["<unk>"]
			elif unk_policy == "zero":
				tokens[i] = 0
			elif unk_policy == "correct":
				corrected = spell_corrector(token)
				if corrected in token2idx:
					tokens[i] = token2idx[token]
				else:
					tokens[i] = token2idx["<unk>"]
	return tokens


def _vectorize_1d(vector, max_length):
	"""
		Convert an array into a zero-padded vector
		according to the max length allowed.

	Args:
		vector: (L)-shaped array
		mex_length: int, max length of vector

	Return:
		a (max_length) numpy.ndarray
	"""
	length = len(vector)
	padded = numpy.zeros(max_length)
	padded[:length] = vector
	return padded


def vectorize_pad(sequence, max_length):
	"""
		Convert a sequence of arrays into a zero-padded sequence
		of arrays according to a maximum length. Useful when
		batching over variable-sized sequences.
	
	Args:
		sequence: (L, D) sequence of arrays
		max_length: int, max length of a sequence

	Return
		a (max_length, D) numpy.ndarray
	"""
	if len(sequence.shape) < 2:
		return _vectorize_1d(sequence, max_length)
	length, dim = sequence.shape
	padded = numpy.zeros((max_length, dim))
	padded[:length, :] = sequence
	return padded


def dummy_preprocess():
	"""
		Dummy preprocessor
	"""
	def preprocess(name, dataset):
		data = [x for x in dataset]
		return data
	return preprocess
	

def twitter_preprocess():
	"""
		ekphrasis-social tokenizer sentence preprocessor.
		Substitutes a series of terms by special coins when called
		over an iterable (dataset)
	"""
	norm = ['url', 'email', 'percent', 'money', 'phone', 'user',
		'time', 'date', 'number']
	ann = {"hashtag", "elongated", "allcaps", "repeated",
		"emphasis", "censored"}
	preprocessor = TextPreProcessor(
		normalize=norm,
		annotate=ann,
		all_caps_tag="wrap",
		fix_text=True,
		segmenter="twitter_2018",
		corrector="twitter_2018",
		unpack_hashtags=True,
		unpack_contractions=True,
		spell_correct_elong=False,
		tokenizer=SocialTokenizer(lowercase=True).tokenize,
		dicts=[emoticons]).pre_process_doc

	def preprocess(name, dataset):
		description = "  Ekphrasis-based preprocessing dataset "
		description += "{}...".format(name)
		data = [preprocessor(x) for x in tqdm(dataset, desc=description)]
		return data

	return preprocess


def remove_wsp_preprocess():
	"""
		Simplest preprocessor ever. Remove whitespaces in a sentence, 
		returning that same sentence as a list of tokens.
	"""
	preprocessor = lambda text: text.split(" ")

	def preprocess(name, dataset):
		description = " Removing whitespaces - preprocessing dataset "
		description += "{}...".format(name)
		data = [preprocessor(x) for x in tqdm(dataset, desc=description)]
		return data

	return preprocess


def remove_stopwords():
	"""
		Removes typical words (nltk stopwords) from the sentences
	"""
	stop_words = set(stopwords.words('english'))
	preprocessor = lambda text: [w for w in text.split(" ") if \
		not w in stop_words]

	def preprocess(name, dataset):
		description = " Removing NLTK stopwords - preprocessing dataset "
		description += "{}...".format(name)
		data = [preprocessor(x) for x in tqdm(dataset, desc=description)]
		return data

	return preprocess


def lemmatizer():
	"""
		Substitutes words by their lemma
	"""
	lemmatizer = WordNetLemmatizer()
	preprocessor = lambda text: [lemmatizer.lemmatize(w) for w in \
		text.split(" ")]

	def preprocess(name, dataset):
		description = " Running NLTK Lemmatizer - preprocessing dataset "
		description += "{}...".format(name)
		data = [preprocessor(x) for x in tqdm(dataset, desc=description)]
		return data

	return preprocess


def stemmer():
	"""
		Substitutes words by their lemma using a PorterStemmer scheme
	"""
	stemmer = PorterStemmer()
	preprocessor = lambda text: [stemmer.stem(w) for w in text.split(" ")]

	def preprocess(name, dataset):
		description = " Running NLTK Porter Stemmer - preprocessing dataset "
		description += "{}...".format(name)
		data = [preprocessor(x) for x in tqdm(dataset, desc=description)]
		return data

	return preprocess


def lemma_nonstop():
	"""
		Combines a lemmatizer and a removal of NLTK stopwords
	"""
	lemmatizer = WordNetLemmatizer()
	stop_words = set(stopwords.words('english'))

	def function(text):
		text = text.split(" ")
		text = [w for w in text if not w in stop_words]
		return [lemmatizer.lemmatize(w) for w in text]

	preprocessor = function

	def preprocess(name, dataset):
		description = " Running NLTK Lemmatizer & removing stopwords "
		description += "- preprocessing dataset "
		description += "{}...".format(name)
		data = [preprocessor(x) for x in tqdm(dataset, desc=description)]
		return data

	return preprocess




