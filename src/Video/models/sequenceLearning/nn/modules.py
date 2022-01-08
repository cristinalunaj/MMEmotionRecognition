import torch

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from src.Video.models.sequenceLearning.environment import DEVICE


class GaussianNoise(nn.Module):
	"""
		Additive Gaussian noise layer

	Args:
		mean: float, mean of the distribution
		stddev: float, standard deviation of the distribution
	"""
	def __init__(self, mean=0, stddev=0):
		super(GaussianNoise, self).__init__()
		self.mean = mean
		self.stddev = stddev

	def forward(self, x):
		"""
			Forward pass - if training, add noise. Otherwise, don't.
		"""
		if self.training:
			noise = Variable(x.data.new(x.size()).normal_(
				self.mean, self.stddev))
			return x + noise
		return x

	def __repr__(self):
		return "{} -- mean = {:.2f}, stddev = {:.2f}".format(
			self.__class__.__name__, self.mean, self.stddev)


class Embed(nn.Module):
	"""
		Definition and initialization tools for an embedding layer,
		thought to be a lokkup table between terms and vectors.

	Args:
		embeddings: np.ndarray 2D lookup table
		num_embeddings: int, length of table
		embedding_dim: int, length of vector embeddings
		trainable: bool, whether to retrain layer
		noise: float, add some gaussian noise to embeddings
		dropout: float, dropout rate
	"""
	def __init__(self, embeddings, num_embeddings, embedding_dim,
		trainable, noise, dropout):
		super(Embed, self).__init__()
		self.embedding = nn.Embedding(num_embeddings=num_embeddings,
			embedding_dim=embedding_dim)
		if embeddings is not None:
			print("Initializing Embedding layer with pretrained weights")
			self.init_embeddings(embeddings, trainable)

		self.dropout = nn.Dropout(dropout)
		self.noise = GaussianNoise(stddev=noise)

	def init_embeddings(self, weights, trainable):
		self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
			requires_grad=trainable)

	def forward(self, x):
		"""
			Forward pass through layer

		Args:
			x: torch.Tensor, input data as a sequence of tokens

		Return:
			a torch.Tensor with that token's embedding
		"""
		embeddings = self.embedding(x)
		if self.noise.stddev > 0:
			embeddings = self.noise(embeddings)
		if self.dropout.p > 0:
			embeddings = self.dropout(embeddings)
		return embeddings


class LSTMEncoder(nn.Module):
	"""
		A simple LSTM-cell-based encoder layer class

	Args:
		input_size: int, input dim
		rnn_size: int, LSTM cell h vector size
		num_layers: int, num of RNN layers
		dropout: dropout rate
		bidirectional: bool, whether layer is bidirectional

	"""
	def __init__(self, input_size, rnn_size, num_layers, dropout,
		bidirectional, batch_first=True):
		super(LSTMEncoder, self).__init__()
		self.lstm = nn.LSTM(input_size=input_size,
			hidden_size=rnn_size,
			num_layers=num_layers,
			dropout=dropout,
			bidirectional=bidirectional,
			batch_first=True)
		self.dropout = nn.Dropout(dropout) # ???
		self.feature_size = rnn_size
		if bidirectional:
			self.feature_size *= 2
		self.batch_first = batch_first
		self.num_layers = num_layers

	@staticmethod
	def last_by_index(outputs, lengths):
		"""
			Index of the last output for every sequence

		Args:
			outputs: 
			lengths:

		Return:

		"""
		idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
			outputs.size(2)).unsqueeze(1)
		return outputs.gather(1, idx).squeeze()

	@staticmethod
	def split_directions(outputs):
		"""
			Tell forward from backward outputs within a
			bidirectional layer

		Args:
			outputs:

		Return:
			a (B, L, H) torch.Tensor from forward LSTM pass
			a (B, L, H) torch.Tensor from backward LSTM pass
		"""
		direction_size = int(outputs.size(-1) / 2)
		forward = outputs[:, :, :direction_size]
		backward = outputs[:, :, direction_size:]
		return forward, backward

	def last_timestep(self, outputs, lengths, bi=False):
		"""
			Get exclusively the last output h_T

		Args:
			outputs:
			lengths:
			bi:

		Return:

		"""
		if bi:
			forward, backward = self.split_directions(outputs)
			last_forward = self.last_by_index(forward, lengths)
			last_backward = backward[:, 0, :]
			return torch.cat((last_forward, last_backward), dim=-1)

		return self.last_by_index(outputs, lengths)

	def forward(self, x, lengths):
		"""
			Forward module pass

		Args:
			x:
			lengths:

		Return:
			a (B, L, H) torch.Tensor of output features for every h_t
			a (B, H) torch.Tensor with the last output h_L
		"""
		packed = pack_padded_sequence(x, list(lengths.data),
			batch_first=self.batch_first)
		out_packed, _ = self.lstm(packed)
		outputs, _ = pad_packed_sequence(out_packed, 
			batch_first=self.batch_first)
		last_outputs = self.last_timestep(outputs, lengths, 
			self.lstm.bidirectional)
		# Force dropout if there s only 1 layer
		if self.num_layers < 2:
			last_outputs = self.dropout(last_outputs)
		return outputs, last_outputs


class SelfAttention(nn.Module):
	"""
		Self-attention layer

	Args:
		attention_size: int, attention vector length
		batch_first: bool, affects tensor ordering of dims
		layers: int, num of attention layers
		dropout: float, dropout rate
		non_linearity: str, activation function
		
	"""
	def __init__(self, attention_size, batch_first=True, layers=1,
		dropout=0, non_linearity="tanh"):
		super(SelfAttention, self).__init__()
		self.batch_first = batch_first
		if non_linearity == "relu":
			activation = nn.ReLU()
		elif non_linearity == "tanh":
			activation = nn.Tanh()
		else:
			raise KeyError("Undefined activation function!")

		modules = []
		for i in range(layers - 1):
			modules.append(nn.Linear(attention_size, attention_size))
			modules.append(activation)
			modules.append(nn.Dropout(dropout))

		modules.append(nn.Linear(attention_size, 1))
		modules.append(activation)
		modules.append(nn.Dropout(dropout))

		self.attention = nn.Sequential(*modules)
		self.softmax = nn.Softmax(dim=-1)

	@staticmethod
	def get_mask(attentions, lengths):
		"""
			Construct mask for padded items from lengths

		Args:
			attentions: torch.Tensor
			lengths: torch.Tensor

		Return:

		"""
		max_len = max(lengths.data)
		mask = Variable(torch.ones(attentions.size())).detach()
		mask = mask.to(DEVICE)
		for i, l in enumerate(lengths.data):
			if l < max_len:
				mask[i, l:] = 0
		return mask

	def forward(self, x, lengths):
		"""
			Forward pass in self-attention.
			Steps:
				- dot product <attention, hidden state>
				- masking by length
				- Weighted sum of scores

		Args:
			x: (B, L, H) torch.Tensor of input sequence vectors
			lengths: 

		Return:
			a (B, H) torch.Tensor of weighted vector values
			a (B, H) torch.Tensor of attention values
		"""
		scores = self.attention(x).squeeze()
		scores = self.softmax(scores)

		mask = self.get_mask(scores, lengths)
		masked_scores = scores * mask
		sums_ = masked_scores.sum(-1, keepdim=True)
		scores = masked_scores.div(sums_)

		weighted = torch.mul(x, scores.unsqueeze(-1).expand_as(x))
		representations = weighted.sum(1).squeeze()

		return representations, scores


