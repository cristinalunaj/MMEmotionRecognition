"""
    Script to plot the average accuracy obtained from the posteriors generated previously running the script: Wav2VecEval.py

	author: Cristina Luna.
	date: 03/2022

	Usage:
		e.g.
		 python3 MMEmotionRecognition/src/Audio/FineTuningWav2Vec/FinalEvaluation.py
		 --dataPosteriors MMEmotionRecognition/data/models/wav2Vec_top_models/FineTuning/data/20211020_094500
		 --trained_model MMEmotionRecognition/data/models/wav2Vec_top_models/FineTuning/trained_models/wav2vec2-xlsr-ravdess-speech-emotion-recognition/20211020_094500
	Options:
        --dataPosteriors: Path with the posteriors
		--trained_model: Path to the fine-tuned model
"""

import os
import sys
import argparse
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import os.path
import pandas as pd
import numpy as np
from transformers import AutoConfig



def extract_acc_from_posteriors(path_posteriors_root, dict_emotions, set = "test"):
    """
    Generate accuracy
    :param path_posteriors_root:[str] Path to the posteriors
    :param: dict_emotions [dict]: Mapping between the names of the emotions and its associated neuron
    :param set:[str] Set to obtain the accuracy (train or test)
    """
    acc = 0
    for fold in range(5):
        path_posteriors_csv = os.path.join(path_posteriors_root, "fold"+str(fold), "posteriors_"+set+".csv")
        df_fold = pd.read_csv(path_posteriors_csv, sep=";", header=0)
        correct = 0
        for i, row in df_fold.iterrows():
            emotion_prediction = np.argmax(row[["embs"+str(i) for i in range(8)]].values)
            emotion_label = dict_emotions[row["emotion"]]
            if(emotion_prediction==emotion_label):
                correct+=1
        acc+= (correct/len(df_fold))

    print("Final AVG accuracy:", str((acc/5)*100))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-d', '--dataPosteriors', type=str, required=True,
                        help='Path with the posteriors')
    parser.add_argument('-trainedModel', '--trained_model', type=str, required=True,
                        help='Path to the trained model')
    args = parser.parse_args()


    trained_model_name = args.trained_model.split("/")[-1]
    model_train_val_test = os.path.join(args.trained_model, "fold0")
    model_train_val_test = os.path.join(model_train_val_test, sorted(os.listdir(model_train_val_test))[-1])

    config = AutoConfig.from_pretrained(os.path.join(model_train_val_test, 'config.json'))

    print("LABELS: ", config.label2id)
    extract_acc_from_posteriors(args.dataPosteriors, config.label2id , set="test")


