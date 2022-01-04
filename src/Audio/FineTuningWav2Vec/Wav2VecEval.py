"""
    Main script to eval the fine-tuned Wav2Vec model.

	author: Cristina Luna. Adapted from the tutorial: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb
	date: 03/2022


	Usage:
		e.g.
		 python3 MMEmotionRecognition/src/Audio/FineTuningWav2Vec/Wav2VecEval.py
		 --data MMEmotionRecognition/data/models/wav2Vec_top_models/FineTuning/data/20211020_094500
		 --fold 0
		 --trained_model MMEmotionRecognition/data/models/wav2Vec_top_models/FineTuning/trained_models/wav2vec2-xlsr-ravdess-speech-emotion-recognition/20211020_094500
		 --out_dir <RAVDESS_dir>/FineTuningWav2Vec2_posteriors
         --model_id jonatasgrosman/wav2vec2-large-xlsr-53-english
	Options:
        --data: Path with the datasets automatically generated with the Fine-Tuning script (train.csv and test.csv)
        --fold: Fold to analyse
		--trained_model: Path to the fine-tuned model
		--out_dir: Path to save the posteriors
        --model_id: Name of the model from the Hugging Face repository to use as baseline. In our case: 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
"""

import os
import sys
import argparse
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import os.path

import librosa
import pandas as pd
from sklearn.metrics import classification_report
from datasets import load_dataset
from transformers import AutoConfig, Wav2Vec2Processor
import torchaudio
import numpy as np
import torch
from src.Audio.FineTuningWav2Vec.Wav2VecAuxClasses import *

def speech_file_to_array_fn(batch):
    """
    Loader of audio recordings. It appends the array of the samples of the recording to the batch dict

    :param batch:[dict] Dict with the data
    :param processor: Global variable with the expected input format of the model
    """
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch


def predict(batch):
    """
    Generate predictions from the model and append to the batch dict the posteriors and predictions

    :param batch:[dict] Dict with the data
                            -speech: input audio recordings [IN]
                            -posteriors: Array with the weight of each of the ourput neuron assigned to each class [OUT]
                            -predicted : Predicted class with the highest posterior [OUT]
    :param: device [str]: Global variable with the device used to load the model and make predictions ('cpu' or 'cuda')

    """
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["posteriors"] = logits.detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

def save_embs(results, path_embs):
    """
    Generate predictions from the model and append to the batch dict the posteriors and predictions

    :param batch:[dict] Dict with the data
                            -speech: input audio recordings [IN]
                            -posteriors: Array with the weight of each of the ourput neuron assigned to each class [OUT]
                            -predicted : Predicted class with the highest posterior [OUT]

    """
    df_posteriors = pd.DataFrame([])
    df_posteriors[["embs"+str(i) for i in range(8)]] = results["posteriors"]
    df_posteriors["name"] = result["name"]
    df_posteriors["emotion"] = result["emotion"]
    df_posteriors["actor"] = result["actor"]
    #Save dataframe
    df_posteriors.to_csv(path_embs, sep=";", header=True, index=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path with the datasets automatically generated with the Fine-Tuning script (train.csv and test.csv)')
    parser.add_argument('-trainedModel', '--trained_model', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('-fold', '--fold', type=int, help='Fold number',
                        default=0)
    parser.add_argument('-out', '--out_dir', type=str, help='Path to save the embeddings extracted from the model',
                        default='./')
    parser.add_argument('-model', '--model_id', type=str,
                        help='Model identificator in Hugging Face library [default: jonatasgrosman/wav2vec2-large-xlsr-53-english]',
                        default='jonatasgrosman/wav2vec2-large-xlsr-53-english')
    args = parser.parse_args()

    #Change if you change the model to choose top weigths
    checkpoints_per_fold = {0: "checkpoint-1300",
                            1: "checkpoint-690",
                            2: "checkpoint-1060",
                            3: "checkpoint-1040",
                            4: "checkpoint-670",
                            }

    trained_model_name = args.trained_model.split("/")[-1]
    path_out_logits = os.path.join(args.out_dir, trained_model_name, "fold"+str(args.fold))
    os.makedirs(path_out_logits, exist_ok=True)

    for set_trainTest in ["train","test"]:
        #Load the data from train.csv or test.csv
        test_dataset = load_dataset("csv", data_files={"test": os.path.join(args.data, "fold"+str(args.fold), set_trainTest+".csv")}, delimiter="\t")["test"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Load the fine-tuned model and its associated processor that has information about the input/output format of the files/outouts in the trianed model
        model_train_val_test = os.path.join(args.trained_model, "fold"+str(args.fold), checkpoints_per_fold[args.fold])
        config = AutoConfig.from_pretrained(os.path.join(model_train_val_test, 'config.json'))
        processor = Wav2Vec2Processor.from_pretrained(args.model_id)
        model = Wav2Vec2ForSpeechClassification.from_pretrained(model_train_val_test).to(device)

        #Load the audio samples
        test_dataset = test_dataset.map(speech_file_to_array_fn)
        # Make predictions on the loaded samples
        result = test_dataset.map(predict, batched=True, batch_size=8)
        label2id_dict = {"Neutral": 0, "Calm": 1, "Happy": 2, "Sad": 3, "Angry": 4, "Fear": 5, "Disgust": 6, "Surprise": 7}

        print("LABELS: ", config.label2id)
        # Get the labels and predictions
        y_true = [config.label2id[name] for name in result["emotion"]]
        y_pred = result["predicted"]

        #Get metrics & save posteriors
        print(classification_report(y_true, y_pred, target_names=list(label2id_dict)))
        #Save embs:
        out_embs = os.path.join(path_out_logits, "posteriors_"+set_trainTest+".csv")
        save_embs(result, out_embs)
