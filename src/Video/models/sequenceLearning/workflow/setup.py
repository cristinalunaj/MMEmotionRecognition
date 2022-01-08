"""
	Setup and execution of the methods in a classical train-val-test
	DL setting
"""
import glob
import os

from os.path import basename
from os.path import join
from os.path import isdir
from os.path import isfile

import src.Video.models.sequenceLearning.nn.models as models
import src.Video.models.sequenceLearning.utils.preprocessors as preproc

from src.Video.models.sequenceLearning.environment import DEVICE, TRAINED_PATH
from src.Video.models.sequenceLearning.dataloaders.datasets import get_data_list, get_dataloaders
from src.Video.models.sequenceLearning.dataloaders.label_transformers import define_label_transformer
from src.Video.models.sequenceLearning.logger.metrics import get_metrics
from src.Video.models.sequenceLearning.logger.prediction import log_evaluation
from src.Video.models.sequenceLearning.logger.prediction import log_inference
from src.Video.models.sequenceLearning.logger.training import Checkpoint, EarlyStop, log_fold_training, log_training, log_K_folds_training
from src.Video.models.sequenceLearning.utils.load_utils import get_pretrained, load_embeddings
from src.Video.models.sequenceLearning.utils.training import class_weights, get_criterion, get_optimizer, get_pipeline

from src.Video.models.sequenceLearning.workflow.trainer import Trainer


def get_output_size(dataloaders):
	"""
		Infer the expected output size and task to perform

	Args:
		dataloaders: dict of torch.dataloader objects

	Return:
		an int, inferred output size
		a str, task to carry out
	"""
	# Find the max amount of unique labels among dataloaders
	nb_labels = 0
	labels_data_type = None
	for partition, dataloader in dataloaders.items():
			labels = list(set(dataloader.dataset.labels))
			if isinstance(labels[0], float):	# regression
				return 1, "regression"

			nb_labels = max(len(labels), nb_labels)

	out_size = 1 if nb_labels == 2 else nb_labels
	task = "binary" if nb_labels == 2 else "multilabel"
	return out_size, task


def define_setup(mode, config, name, datasets, monitor="val", pretrained=None,
	finetune=False, label_transformer=None, disable_cache=False, trained_path = ""):
	"""
		Prepare the pipeline for a typical DL workflow

	Args:
		mode: str, to choose between train, test or inference
		config:	dict, experiment parameters
		name:	str, name of the experiment
		datasets: dict, data for every data partition (X, y)
		monitor: str, partition to watch on learning time 
		pretrained: str, path to pretrained model and conf files
		finetune: bool, whether to finetune pretrained model
		label_transformer: Label transform function
		disable_cache: Whether to activate data cache

	Return:
		a Trainer object
	"""
	pretrained_model, pretrained_config = None, None
	if pretrained:
		pretrained_model, pretrained_config = get_pretrained(pretrained, trained_path)

	word2idx, embeddings = None, None
	if config.get("embeddings_file", None):	# Precomputed embeddings
		word2idx, idx2word, embeddings = load_embeddings(config)

	try:
		preprocessor = getattr(preproc, config["preprocessor"])
	except TypeError:
		preprocessor = preproc.dummy_preprocess	# Do nothing on input

	loaders = get_dataloaders(datasets,
		batch_size=config["batch_size"],
		data_type=config["data_type"],
		name=name,
		preprocessor=preprocessor(),
		label_transformer=label_transformer,
		word2idx=word2idx,
		config=config)

	output_size, task = get_output_size(loaders)
	weights = None
	if task != "regression" and "train" in loaders:
		weights = class_weights(loaders["train"].dataset.labels,
			to_pytorch=True).to(DEVICE)

	arch = getattr(models, config["model_name"])
	model_params = config.get("model_params", None)
	model = arch(out_size=output_size,
		embeddings=embeddings,
		input_size=config["input_size"],
		pretrained=pretrained_model,
		finetune=finetune,
		model_params=model_params).to(DEVICE)

	criterion = get_criterion(task, weights)
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = get_optimizer(parameters, lr=config["lr"],
		weight_decay=config["weight_decay"])
	pipeline = get_pipeline(task, criterion)

	metrics, monitor_metric, task_mode = get_metrics(task)
	model_dir = None
	if pretrained:
		model_dir = os.path.join(TRAINED_PATH, "TL")

	checkpoint, early_stopping = None, None
	if mode == "train":
		checkpoint = Checkpoint(name=name, 
			model=model, 
			model_conf=config,
			monitor=monitor, 
			keep_best=True, 
			timestamp=True, 
			scorestamp=True, 
			metric=monitor_metric, 
			mode=task_mode, 
			base=config["base"],
			model_dir=model_dir)

		if config["patience"] != -1:
			early_stopping = EarlyStop(metric=monitor_metric,
				mode=task_mode,
				monitor=monitor,
				patience=config["patience"],
				min_change=config["min_change"])

	return Trainer(model=model,
		loaders=loaders,
		task=task,
		config=config,
		optimizer=optimizer,
		pipeline=pipeline,
		metrics=metrics,
		checkpoint=checkpoint,
		early_stopping=early_stopping)	

def execute(mode, params, pretrained=None, finetune=False, kfolds=None):
	"""
		Complete a full pass on a given mode.

	Args:
		mode: str, to choose between train, test or inference
		pretrained: str, path to a pretrained model and conf file
		finetune: bool, whether to retrain whole model or last layer(s)
		kfolds: int, number of folds in Cross Validation

	"""
	model_config = params
	task_name = model_config["name"]
	desc_name = ""
	try:
		trained_path = model_config["TRAINED_PATH"]
	except:
		trained_path = ""
	try:
		out_path = model_config["OUT_PATH"]
	except:
		out_path = ""

	if mode != "train":
		assert pretrained, "Missing a pretrained model, perhaps?"

	if kfolds:
		results = []
		for fold in range(kfolds):
			print("*" * 30)
			print("[FOLD {} OUT OF {}]".format(fold + 1, kfolds))
			print("*" * 30)
			fold_desc = "-fold_{}_outof_{}".format(fold + 1, kfolds)

			pt_name = None
			if pretrained:
				pt_is_file = isfile(join(TRAINED_PATH, pretrained + 'model'))
				pt_is_dir = isdir(pretrained)
				assert pt_is_dir or isinstance(pretrained, str)

				if pt_is_dir:
					#lOOK FOR THE BEST MODEL
					pt_name = sorted(glob.glob(os.path.join(
						pretrained, task_name + '*' + fold_desc + '*.model')))[-1] #TRAINED_PATH(donde pretrained)
					pt_name = pt_name.split('/')[-1].replace('.model', '')
					print("LOADING MODEL: ", pt_name)
					if finetune:
						desc_name += "-FT-" + str_finetune
				else:
					pt_name = basename(pretrained)
					desc_name += '-' + pt_name
					if finetune:
						desc_name += "-FT-" + str(finetune)


			dataset_name = params["name"]
			if mode == "train":
				datasets = {"train": get_data_list(dataset_name, 
					key="training" + fold_desc)}
				try:
					datasets["val"] = get_data_list(dataset_name,
						key="validation" + fold_desc)
					monitoring = "val"
				except FileNotFoundError:
					print('[FIXED NUMBER OF EPOCHS IN TRAINING]')
					monitoring = "train"
				label_transformer = define_label_transformer(datasets["train"])
			elif mode=="test":

				datasets = {
					"test": get_data_list(dataset_name,key="test" + fold_desc)
				}
				label_transformer = define_label_transformer(datasets["test"])
				monitoring = None
			else: #INFERENCE
				# Data to extract posteriors
				if (params["inference_data"] == "train"):
					key = "training"
				elif (params["inference_data"] == "test"):
					key = "test"
				elif (params["inference_data"] == "val"):
					key = "validation"
				datasets = {
					params["inference_data"]: get_data_list(dataset_name, key=key + fold_desc)
				}
				label_transformer = define_label_transformer(datasets[list(datasets.keys())[0]])
				monitoring = None

			experiment_setup = define_setup(mode=mode,
				config=model_config,
				name=(task_name + desc_name + fold_desc),
				datasets=datasets,
				monitor=monitoring,
				pretrained=pt_name,
				finetune=finetune,
				label_transformer=label_transformer,
				disable_cache=model_config.get("disable_cache", False),
				trained_path=trained_path)

			if mode == "train":
				experiment_setup.train(model_config["epochs"])
				log_training(experiment_setup, task_name, desc_name + fold_desc)
				results.append(log_fold_training(experiment_setup))
			elif mode == "test":
				experiment_setup.eval()
				log_evaluation(experiment_setup, task_name, desc_name + fold_desc)
				results.append(log_fold_training(experiment_setup))
			else:
				experiment_setup.inference()
				log_inference(experiment_setup, task_name, "-" + desc_name + fold_desc, out_path)

		if mode != "inference":
			log_K_folds_training(experiment_setup,
				task_name, desc_name, results)


	else:
		if pretrained:
			pt_name = basename(pretrained)
			desc_name += '-' + pt_name
			if finetune:
				desc_name += "-FT-" + str(finetune)

		dataset_name = params["name"]
		if mode == "train":
			datasets = {"train": get_data_list(dataset_name, key="training")}
			try:
				datasets["val"] = get_data_list(dataset_name, key="validation")
				monitoring = "val"
			except FileNotFoundError:
				print('[FIXED NUMBER OF EPOCHS IN TRAINING]')
				monitoring = "train"
			label_transformer = define_label_transformer(datasets["train"])

		elif mode== "test":
			datasets = {
			"test": get_data_list(dataset_name, key="test")
			}
			label_transformer = define_label_transformer(datasets["test"])
			monitoring = None
		elif mode == "inference":
			#Data to extract posteriors
			if(params["inference_data"]=="train"):
				key = "training"
			elif(params["inference_data"]=="test"):
				key = "test"
			elif (params["inference_data"] == "val"):
				key = "validation"
			datasets = {
				params["inference_data"]: get_data_list(dataset_name, key=key)
			}
			label_transformer = define_label_transformer(datasets[list(datasets.keys())[0]])
			monitoring = None

		experiment_setup = define_setup(mode=mode,
			config=model_config,
			name=task_name,
			datasets=datasets,
			monitor=monitoring,
			pretrained=pretrained,
			finetune=finetune,
			label_transformer=label_transformer,
			disable_cache=model_config.get("disable_cache", False),
			trained_path = trained_path)

		if mode == "train":
			experiment_setup.train(model_config["epochs"])
			log_training(experiment_setup, task_name, desc_name)

		elif mode == "test":
			experiment_setup.eval()
			log_evaluation(experiment_setup, task_name, desc_name)

		else:#inference
			experiment_setup.inference()
			log_inference(experiment_setup, task_name, desc_name)








