# AUs + Sequential Models 
Modified version from the code of the repository: https://github.com/affectivepixelsteam/sequenceLearning


## 1. Create the configuration file
    
Create the configuration file in sequenceLearning/conf/RAVDESS_AUs.json. This file specifies information about the input format
and parameters of the model such as number of layer, type of model, batch size, learning rate...
Some important parameters are:
    
    "data_type": "pathCSV". Indicate that we will use csv files to load the samples
    ""model_name": "Sequence1Modal", ##Model to use (See MMEmotionRecognition/src/Video/models/sequenceLearning/nn/models.py)
    "input_size": 35   ## Number of columns/attributes at the input
    "dim": 50  ##Number of neurons in each layer
    "layers": 2,  ##Number of LSTM layers
    "dropout": 0.3, 
    "bidirectional": true  ## if True, we are using bi-LSTMs, else, LSTMs

## 2. Create the datasets
Firt to train the model, we need to distribute the data. To create the datasets, we run:

    python3 src/Video/models/sequenceLearning/frontend/RAVDESS_AUs/frontend_ravdess_5CV.py
    --AUs_dir <RAVDESS_dir>/processed_AUs
    --out_dir MMEmotionRecognition/src/Video/models/sequenceLearning/datasets/RAVDESS_AUs

## 3. Run set-up

Train the sequential model with the parameters specified in RAVDESS_AUs.json. 
*Note: The generated model will be saved in: MMEmotionRecognition/src/Video/models/sequenceLearning/out
(5CV)

    python3 MMEmotionRecognition/src/Video/models/sequenceLearning/workflow/run.py train MMEmotionRecognition/src/Video/models/sequenceLearning/conf/RAVDESS_AUs.json --kfolds 5


## 4. Extract posteriors
Once the model is trained, we can extract the posteriors running:

    python3 workflow/run.py inference
    MMEmotionRecognition/src/Video/models/sequenceLearning/conf/RAVDESS_AUs_posteriors.json
    --kfolds 5
    --pretrained MMEmotionRecognition/src/Video/models/sequenceLearning/out/trained

*Notice that you should mofify the RAVDESS_AUs_posteriors.json to extract the embeddings from the training and from the validation
by changing the 'inference_data' parameter (See example below)

### Example RAVDESS_AUs_posteriors.json:
    {
	"name": "RAVDESS_AUs", //name of the task
	"data_type": "pathCSV",
	"model_name": "Sequence1Modal",
	"inference_data": "train", //Data to extract posteriors [Options: train, val]
	"OUT_PATH": "<RAVDESS_dir>/FUSION/wav2Vec_AUs/BiLSTM_AUS/posteriors", //Path to save the posteriors
	"TRAINED_PATH": "MMEmotionRecognition/src/Video/models/sequenceLearning/out/trained", //Path with the trained models
	"input_size": 35, 
	"model_params": // Parameters of the model saved in trained models 
	{"encoder":
		{
			"dim": 50,
			"layers": 2,
			"dropout": 0.3,
			"bidirectional": true

		},
	"attention":
		{
			"layers":2,
			"dropout": 0.3,
			"activation": "tanh",
			"context": false
		}
	},
	"preprocessor": null,
	"batch_size": 64,
	"lr": 1e-3,
	"weight_decay": 0.0,
	"patience": 30,
	"min_change": 0.0,
	"epochs": 300,
	"base": 0.0,
	"clip_norm": 1,
	"seed": 2020,
	"disable_cache": true
    }
