import torch
import torch.nn as nn

from src.Video.models.sequenceLearning.environment import DEVICE
from src.Video.models.sequenceLearning.nn.modules import Embed, LSTMEncoder, SelfAttention

from torchvision.models import densenet121


class FeatureExtractor(nn.Module):
	"""
		Feature extraction initialization part of a model, based on biLSTM
		layer(s) and attention layer(s).

	Args:
		embeddings: np.ndarray of pretrained embeddings
		input_size: int, embeddings or input array s size
		embed_params: dict, set of embeddings layer options
			- dim: int, embeddings length
			- finetune: bool, whether to train layer
			- noise: float, white noise variance to add
			- dropout: float, dropout rate
		encoder_params:
			- size: int, encoding vector's shape
			- layers: int, num of encoder layers
			- dropout: float, dropout rate
			- bidirectional: bool, whether it is bidirectional
		attention_params:
			- layers: int, number of attention layers
			- dropout: float, dropout rate
			- activation: str, activation function
			- context: bool, whether to normalize attention over context
	"""
	def __init__(self, embeddings=None, input_size=0, embed_params=None,
		encoder_params=None, attention_params=None, **kwargs):
		super(FeatureExtractor, self).__init__()
		if embeddings is not None:
			dim, embed_finetune, noise, dropout = self._get_params(
				embed_params)
			self.embedding = Embed(
				num_embeddings=embeddings.shape[0],
				embedding_dim=embeddings.shape[1],
				embeddings=embeddings,
				trainable=embed_finetune,
				noise=noise,
				dropout=dropout)
			encoder_input_size = embeddings.shape[1]
		else:
			encoder_input_size = input_size

		encoder_dim, num_layers, dropout, bi = self._get_params(
			encoder_params)
		self.encoder = LSTMEncoder(input_size=encoder_input_size,
			rnn_size=encoder_dim,
			num_layers=num_layers,
			bidirectional=bi,
			dropout=dropout)
		self.feature_size = self.encoder.feature_size

		if attention_params:
			layers, dropout, activation, context = self._get_params(
				attention_params)
			self.attention_context = context
			attention_size = self.feature_size
			if context:
				context_size = self.encoder.feature_size
				attention_size += context_size

			self.attention = SelfAttention(
				attention_size=attention_size,
				layers=layers,
				dropout=dropout,
				non_linearity=activation)

	@staticmethod
	def _get_params(params):
		"""
			Retrieve the values as specified in a params dict
		"""
		return (val for val in params.values())		

	@staticmethod
	def _mean_pooling(x, lengths):
		sums = torch.sum(x, dim=1)
		lens_ = lengths.view(-1, 1).expand(sums.size(0), sums.size(1))
		means = sums / lens_.float()
		return means

	def forward(self, x, lengths):
		"""
			Forward pass
		"""
		if hasattr(self, "embedding"):
			x = self.embedding(x)

		outputs, last_output = self.encoder(x.float(), lengths)
		attentions = None
		representations = last_output

		if hasattr(self, "attention"):
			if self.attention_context:
				context = self._mean_pooling(outputs, lengths)
				context = context.unsqueeze(1).expand(-1, outputs.size(1), -1)
				outputs = torch.cat([outputs, context], -1)
			representations, attentions = self.attention(outputs, lengths)
			if self.attention_context:
				representations = representations[:, :context.size(-1)]

		return representations, attentions


class SequenceSorter:
	"""
		Sort batch data and labels by sequence length
	
	Args:
		lengths: nn.Tensor, lengths of the data sequences

	Return:
		a nn.Tensor or sorted lengths
		a callable method that sorts iterable items
		a callable method that unsorts iterable items to original
			order
	"""
	@staticmethod
	def _sort_by(lengths):
		batch_size = lengths.size(0)
		sorted_lengths, sorted_idx = lengths.sort()
		_, original_idx = sorted_idx.sort(0, descending=True)
		reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()
		reverse_idx = reverse_idx.to(DEVICE)
		sorted_lengths = sorted_lengths[reverse_idx]

		def sort(iterable):
			if len(iterable.shape) > 1:
				return iterable[sorted_idx.data][reverse_idx]
			return iterable

		def unsort(iterable):
			if len(iterable.shape) > 1:
				return iterable[reverse_idx][original_idx][reverse_idx]
			return iterable

		return sorted_lengths, sort, unsort


class Sequence1Modal(nn.Module, SequenceSorter):
	"""
		Sequential model aimed at uni-modal problems,
		based on a encoder-decoder structure. 

		The encoder is conformed by a stack of (bi)LSTM cells.
		The decoder is made by a self-attention layer and a Linear
		classifier.

		Args:
		out_size: int, output layer's shape
		embeddings: np.ndarray of pretrained embeddings
		embed_dim: int, embeddings size
		pretrained: str, path to pretrained model and conf files
		finetune: bool, whether to finetune pretrained model
		embed_params: dict, set of embeddings layer options
			- dim: int, embeddings length
			- finetune: bool, whether to train layer
			- noise: float, white noise variance to add
			- dropout: float, dropout rate
		encoder_params:
			- size: int, encoding vector's shape
			- layers: int, num of encoder layers
			- dropout: float, dropout rate
			- bidirectional: bool, whether it is bidirectional
		attention_params:
			- layers: int, number of attention layers
			- dropout: float, dropout rate
			- activation: str, activation function
			- context: bool, whether to normalize attention over context
		NOTE: all *_params required to follow above key ordering
	"""
	def __init__(self, out_size, input_size, model_params, 
		embeddings=None, pretrained=None, finetune=False, **kwargs):
		super(Sequence1Modal, self).__init__()
		embed_params = model_params.get("embeddings", None)
		encoder_params = model_params["encoder"]
		attention_params = model_params.get("attention", None)

		self.feature_extractor = FeatureExtractor(
			embeddings=embeddings,
			input_size=input_size,
			embed_params=embed_params,
			encoder_params=encoder_params,
			attention_params=attention_params)
		self.feature_size = self.feature_extractor.feature_size
		self.linear = nn.Linear(in_features=self.feature_size,
			out_features=out_size)
		
		if pretrained:
			print("[LOADING PRETRAINED MODEL]")
			self.feature_extractor = pretrained.feature_extractor
			if embed_params:
				noise = embed_params["noise"]
				self.feature_extractor.embedding.noise.stddev = noise
				dropout = embed_params["dropout"]
				self.feature_extractor.embedding.dropout.p = dropout

			encoder_dropout = encoder_params["dropout"]
			self.feature_extractor.encoder.dropout.p = encoder_dropout

			if attention_params:
				for module in self.feature_extractor.attention.attention:
					if isinstance(module, nn.Dropout):
						module.p = attention_params["dropout"]

			self.linear = pretrained.linear

			if not finetune:
				for param in self.feature_extractor.parameters():
					param.requires_grad = False

	def forward(self, x, lengths):
		"""
			Forward pass through the network

		Args:
			x: nn.Tensor, sequences of data items
			lengths: nn.Tensor, sequences' lenghts

		Return:
			a (B, out_size) nn.Tensor, class logits
			a (B, encoder_dim) nn.Tensor, attention values
		"""
		lengths, sort, unsort = self._sort_by(lengths)
		x = sort(x)
		representations, attentions = self.feature_extractor(x, lengths)

		representations = unsort(representations)
		if attentions is not None:
			attentions = unsort(attentions)
		logits = self.linear(representations)
		return logits, attentions
	

class Densenet121(nn.Module):
	"""
		Densenet-121 torch implementation from the
		torch's model hub.
	"""
	def __init__(self, out_size, input_size=0,
		model_params=None, embeddings=None, pretrained=None, 
		finetune=False, **kwargs):
		super(Densenet121, self).__init__()
		self.basemodel = densenet121(
			pretrained=True,
			progress=True)
		feature_size = self.basemodel.classifier.in_features
		self.basemodel.classifier = nn.Linear(
			in_features=feature_size,
			out_features=out_size)
		self.sigmoid = nn.Sigmoid()

		if pretrained:
			print("[LOADING PRETRAINED MODEL]")
			self.basemodel = pretrained.basemodel

	def forward(self, x, lengths=None):
		"""
			Forward pass
		Args:
			x: (B, 3, 224, 224) nn.Tensor, image tensor
			lengths: None, required for compatibility
		Return:
			a (B, out_size) nn.Tensor, model output
			a None object, needed for compatibility
		"""
		x = x.float()
		h = self.basemodel(x)
		logits = self.sigmoid(h)
		return logits, None


class MLP(nn.Module):
	"""
		Simple single layered MLP
	"""
	def __init__(self, out_size, input_size,
		model_params=None, embeddings=None, pretrained=None,
		finetune=False, **kwargs):
		super(MLP, self).__init__()
		self.hidden = nn.Linear(input_size, out_size)
		self.sigmoid = nn.Sigmoid()

		if pretrained:
			print("[LOADING PRETRAINED MODEL]")
			self.hidden = pretrained.hidden

	def forward(self, x, lengths=None):
		x = x.float()
		y_hat = self.hidden(x)
		y_hat = self.sigmoid(y_hat)
		return y_hat, None


