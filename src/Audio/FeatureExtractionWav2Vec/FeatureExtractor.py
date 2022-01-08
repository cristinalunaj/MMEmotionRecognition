"""
    Script to extract the features at the output of the feature encoder (CNNs) of the Wav2Vec2.0 model

	author: Cristina Luna.
	date: 03/2022

	Usage:
		e.g.
		 python3 MMEmotionRecognition/src/Audio/FeatureExtractionWav2Vec/FeatureExtractor.py
		 --data MMEmotionRecognition/data/models/wav2Vec_top_models/FineTuning/data/20211020_094500
		 --model_id jonatasgrosman/wav2vec2-large-xlsr-53-english
		 --out_dir <RAVDESS_dir>/FineTuningWav2Vec2_embs512
	Options:
        --data: Path with the datasets automatically generated with the Fine-Tuning script (train.csv and test.csv)
		--model_id: Path to the baseline model to extract the features from
		--out_dir: Path to save the embeddings
"""



import argparse
import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import librosa
import pandas as pd
from datasets import load_dataset
from transformers import Wav2Vec2Processor
import torchaudio
import numpy as np
import torch
from src.Audio.FineTuningWav2Vec.Wav2VecAuxClasses import *

def speech_file_to_array_fn(batch):
    """
        Loader of audio recordings. It appends the array of the samples of the recording to the batch dict

        :param batch:[dict] Dict with the data
        :param processor[Wav2Vec2Processor]: Global variable with the expected input format of the model
        """
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch


def extract_features(batch, device, model, processor):
    """
        Generate features from the model and append to the batch dict the posteriors and predictions

        :param batch:[dict] Dict with the data
                                -speech: input audio recordings [IN]
                                -predicted : Embeddings extracted from the last layer of the feature encoder (CNNs block)
        :param: device [str]: Device used to load the model and make predictions ('cpu' or 'cuda')
        :param: model [Wav2Vec2Model]: Model to extract the embeddings from.
        :param processor[Wav2Vec2Processor]: Information of the expected input format of the model

    """
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask)
        #feats = processor.feature_extractor(input_values)

    batch["predicted"] = logits["extract_features"]
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path with the datasets automatically generated with the Fine-Tuning script (train.csv and test.csv)')
    parser.add_argument('-out', '--out_dir', type=str, help='Path to save the embeddings extracted from the model', default='./')
    parser.add_argument('-model', '--model_id', type=str, help='Model identificator in Hugging Face library [default: jonatasgrosman/wav2vec2-large-xlsr-53-english]',
                        default='jonatasgrosman/wav2vec2-large-xlsr-53-english')

    args = parser.parse_args()

    test_dataset = load_dataset("csv", data_files={"test": os.path.join(args.data, "train.csv")}, delimiter="\t")["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(args.model_id)
    model = Wav2Vec2Model.from_pretrained(args.model_id).to(device)
    test_dataset = test_dataset.map(speech_file_to_array_fn)
    cols = ["embs" + str(i) for i in range(512)]
    for row in test_dataset:
        result = extract_features(row, device, model, processor)
        df_aux = pd.DataFrame(result['predicted'].cpu().numpy().reshape(-1, 512), columns=cols)
        df_aux.to_csv(os.path.join(args.out_dir, result["name"]+".csv"), sep=";", index=False, header=True)






