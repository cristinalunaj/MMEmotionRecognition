from os.path import join
from torch.utils.data import DataLoader

from src.Video.models.sequenceLearning.dataloaders.dataloaders import PathDataset, PathDatasetCSV
from src.Video.models.sequenceLearning.dataloaders.dataloaders import TokenDataset
from src.Video.models.sequenceLearning.environment import DATA_DIR


def get_data_list(dataset_name, key):
	"""
		From the name of the dataset, retrieve training and validation
		lists from files.

		For every dataset, a folder within `datasets` must be created,
		keeping the same name as in config["name"] in a configuration
		json file. Within that directory, two lists are expected:
			- training.txt
			- validation.txt

		Each of these files have the samples in the format:
			<sample #>	<label>	<sample as sentence or path to values>
		with fields separated by tabs.

	Args:
		dataset_name: name of the dataset. Folder to direct to.
		key: partition name of data list [training/validation/test]

	Return:
		tuple of (samples, labels) for training and validation partitions
	"""
	with open(join(DATA_DIR, dataset_name, key + '.txt'), 'r') as file:
		data = [line.strip().split('\t') for line in file.readlines()]
	X = [d[2] for d in data]
	y = [d[1] for d in data]
	names = [d[3] for d in data]
	try:
		y = [float(i) for i in y]
	except:
		pass
	return (X, y, names)


def get_dataloaders(datasets, batch_size, data_type, preprocessor=None,
	name=None, label_transformer=None, **kwargs):
	"""
		Prepare, define and retrieve appropiate data loaders
		for the dataset partitions.

	Args:
		datasets: dict, {key:partition, val:dict of X, y lists}
		batch_size: int, batch size
		data_type: str, [token|path]
		preprocessor: callable, sample preprocessor 
		name: str, name of the dataset
		label_transformer: LabelTransformer

	Return:
		a dict of {partition: partition_dataloader}
	"""
	loaders = {}
	if data_type == "token":
		try:
			word2idx = kwargs["word2idx"]
			assert word2idx
		except KeyError as err:
			raise err("Word2idx dictionary undefined!")

		try:
			unk_policy = kwargs["config"]["unk_policy"]
		except KeyError:
			raise KeyError("Policy for OOV tokens undefined!")

		try:
			spell_corrector = kwargs["config"]["spell_corrector"]
		except KeyError:
			raise KeyError("Spell corrector undefined!")

		for partition, data in datasets.items():
			_name = name + "_{}".format(partition.lower())
			print("  Building TOKENIZED dataset {}...".format(_name))
			dataset = TokenDataset(X=data[0],
				labels=data[1],
				name=_name,
				samples=data[2],
				word2idx=word2idx,
				preprocess=preprocessor,
				label_transformer=label_transformer,
				unk_policy=unk_policy,
				spell_corrector=spell_corrector)
			shuffle = True if partition == "train" else False
			loaders[partition] = DataLoader(dataset, batch_size,
				shuffle=shuffle, drop_last=True)
	
	elif data_type == "path":
		for partition, data in datasets.items():
			_name = name + "_{}".format(partition.lower())
			print("  Builting PATH dataset {}...".format(_name))
			dataset = PathDataset(X=data[0],
				labels=data[1],
				name=_name,
				samples=data[2],
				preprocess=preprocessor,
				label_transformer=label_transformer,
				)#disable_cache=disable_cache
			shuffle = True if partition == "train" else False
			loaders[partition] = DataLoader(dataset, batch_size,
				shuffle=shuffle, drop_last=True)
	elif data_type == "pathCSV":
		for partition, data in datasets.items():
			_name = name + "_{}".format(partition.lower())
			print("  Builting PATH CSV dataset {}...".format(_name))
			dataset = PathDatasetCSV(X=data[0],
				labels=data[1],
				name=_name,
				samples=data[2],
				preprocess=preprocessor,
				label_transformer=label_transformer,
				sep=";") #disable_cache=disable_cache,
			shuffle = True if partition == "train" else False
			loaders[partition] = DataLoader(dataset, batch_size,
				shuffle=shuffle, drop_last=False)

	# elif data_type == "image":
	# 	for partition, data in datasets.items():
	# 		_name = name + "_{}".format(partition.lower())
	# 		print("  Building IMAGE dataset {}...".format(_name))
	# 		dataset = ImageDataset(X=data[0],
	# 			labels=data[1],
	# 			name=_name,
	# 			samples=data[2],
	# 			preprocess=preprocessor,
	# 			label_transformer=label_transformer,
	# 			disable_cache=disable_cache)
	# 		shuffle = True if partition == "train" else False
	# 		loaders[partition] = DataLoader(dataset, batch_size,
	# 			shuffle=shuffle, drop_last=True)

	else:
		raise KeyError("Invalid data type. Valid values: [token|path]")

	return loaders

	
