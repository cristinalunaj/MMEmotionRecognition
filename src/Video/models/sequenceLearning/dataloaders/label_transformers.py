import pickle

def define_label_transformer(dataset):
	labels = dataset[1]
	try:
		labels = [float(x) for x in labels]
		if isinstance(labels[0], float):
			return None		# regression task, no processor
		else:
			raise ValueError("Data type undefined!")	
	except:
		return categorical_transformer(labels)


def categorical_transformer(labels):
	"""
		Transfomer to assign indices to categories from a set of labels.

	Args:
		labels:

	Return:
		a LabelTransformer fit to labels
	"""
	label_map = {label:idx for idx, label in enumerate(
		sorted(list(set(labels))))}
	inv_label_map = {v: k for k, v in label_map.items()}
	print("label_map: ", label_map)
	return LabelTransformer(label_map, inv_label_map)
	

class LabelTransformer:
	"""
		Class creating a custom mapping between labels and indices.

	Args:
		mapping: dict whose key = labels, values = indices
		inv_map: dict whose key = indices, values = labels
	"""
	def __init__(self, mapping, inv_map=None):
		self.map = mapping
		self.inv_map = inv_map

		if self.inv_map is None:
			self.inv_map = {v: k for k, v in self.map.items()}

	def transform(self, label):
		return self.map[label]

	def inverse(self, index):
		return self.inv_map[index]

	def num_classes(self):
		return len(self.map.keys())

	def save_labelTransformer(self, path2save):
		with open(path2save, 'wb') as f:
			pickle.dump(self.map, f, pickle.HIGHEST_PROTOCOL)

	def load_labelTransformer(self, path2load):
		with open(path2load, 'rb') as f:
			self.map = pickle.load(f)
		self.inv_map = {v: k for k, v in self.map.items()}