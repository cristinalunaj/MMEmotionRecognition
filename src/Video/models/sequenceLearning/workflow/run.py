"""
	Train, validate and test your models from this script.

	Operating as a sort of front-end, this script collects
	the information from both a configuration file and
	command line arguments in order to provide the rest of the
	pipeline with everything that is required to proceed.

	Usage:
		python3 workflow/run.py <mode> <conf> [--options]

	Options:
		--pretrained	Name of the model to load
		--finetune	Layer from which to retrain a pretrained model
		--kfolds	Number of K folds in a Cross-Validation scheme
		-h, --help	Display this message
"""

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../../')

import argparse
import json
import numpy
import os
import random
import torch

from datetime import datetime

import src.Video.models.sequenceLearning.workflow.setup as setup


def set_random_seed(seed=1234):
	"""
		Fix the random seed for upcoming operations
	"""
	print(" INICIALIZANDO SEMILLA RANDOM!!! ")
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	#torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def parse_command_line_args():
	parser = argparse.ArgumentParser(
		description='Torch-based DL workflow and workbench',
		formatter_class=argparse.RawTextHelpFormatter)

	# Mandatory arguments
	parser.add_argument("mode",
		type=str,
		choices=['train', 'test', 'inference'],
		help="Workflow scheme [train | test | inference]")
	parser.add_argument("conf",
		type=str,
		help="Path to configuration file defining an experiment")

	# Options
	parser.add_argument("--pretrained",
		type=str,
		help="Name shared by the .model and .conf files of a pretrained model")
	parser.add_argument("--finetune",
		action="store_true",
		help="Whether to limit retraining to last layer")
	parser.add_argument("--kfolds",
		type=int,
		default=None,
		help="Number of K folds in Cross-Validation")

	args = parser.parse_args()
	if args.kfolds and args.kfolds < 2:
		parser.error('K-folds requires at least K=2 folds')
	return parser.parse_args()


def display_params(args, params):
	"""
		Display configuration paramenters on terminal
	"""
	print("SCRIPT: " + os.path.basename(__file__))
	print('Options...')
	for arg in vars(args):
		print('  ' + arg + ': ' + str(getattr(args, arg)))
	print('-' * 30)

	print('Config-file params...')
	for key, value in params.items():
		print('  ' + key + ': ' + str(value))
	print('-' * 30)


def load_json(json_file):
	"""
		Load parameters from a configuration json file
	"""
	with open(json_file, 'r') as file:
		jdata = json.load(file)
	return jdata


def _main_():
	args = parse_command_line_args()
	params = load_json(args.conf)
	display_params(args, params)
	init_time = datetime.now()
	print("Process initiated at time: ", init_time.strftime("%H:%M:%S"))
	set_random_seed(2020)

	setup.execute(mode=args.mode,
		params=params,
		pretrained=args.pretrained,
		finetune=args.finetune,
		kfolds=args.kfolds)

	print("Finished at time: ", datetime.now().strftime("%H:%M:%S"))

if __name__ == "__main__":
	_main_()













