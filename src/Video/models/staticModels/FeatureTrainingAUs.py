"""
    Script to train the Static Models using the average AUs of each video

    author: Cristina Luna.
	date: 03/2022

	Usage:
		e.g.
		 python3 MMEmotionRecognition/src/Video/models/FeatureTraining_avg.py
		 --AUs_dir <RAVDESS_dir>/videos
		 --model_number 1
		 --param 0.01
		 --type_of_norm 0
		 --out_dir <RAVDESS_dir>/posteriors_AUs_SVC_C001
	Options:
         --AUs_dir Path with the embeddings to train/test the models
		 --model_number: Number to identify the model to train and test [1-11]: 1-SVC / 2- Logistic Regression / 3- ridgeClassifier /4-perceptron / 5-NuSVC / 6-LinearSVC / 7-knn / 8-NearestCentroid / 9- DecrissionTree / 10- RandomForest / 11 - MLP')
		 --param: Parameter of the model: C in SVC / C in Logistic Regression / alpha in ridgeClassifier / alpha in perceptron / nu in NuSVC / C in LinearSVC / k in knn / None in NearestCentroid / min_samples_split in DecrissionTree / n_estimators in RandomForest / hidden_layer_sizes in MLP
		 --type_of_norm: Normalizaton to apply: '0-MinMax Norm / 1-Standard Norm / 2- No apply normalization [default: 2]
		 --out_dir : Path to save the posteriors of the trained models
"""

import os.path, os, sys
import argparse
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

import pandas as pd
import numpy as np
from src.Audio.FeatureExtractionWav2Vec.FeatureTraining import get_classifier, extract_posteriors, generate_train_test, clean_df
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_AUs_avg(embs_path, avg_embs_path):
    """
           Save Dataframes with only the AUs columns
            :param embs_path:[str] Path where the processed AUs where saved after the extraction process of OpenFace. This embeddings has dimension (35,timesteps)
            :param: avg_embs_path [str]:  Path to save the average AUs calculated collapsing the timesteps. This embeddings has dimension (35,1)
        """
    if (os.path.exists(avg_embs_path)):
        X_total = pd.read_csv(avg_embs_path, sep=";", header=0)
    else:
        X_total = pd.DataFrame([])
        for video_embs in os.listdir(embs_path):
            embs_df = pd.read_csv(os.path.join(embs_path, video_embs), sep=";", header=0)

            aux_df = pd.DataFrame([embs_df.mean()], columns=embs_df.columns)
            aux_df["name"] = video_embs.split(".")[0]
            aux_df["path"] = video_embs.split(".")[0]
            aux_df["index"] = 0
            X_total = X_total.append(aux_df)
        X_total.to_csv(avg_embs_path, sep=";", header=True, index=False)

    X_total = X_total.rename(columns={"name":"video_name"})
    X_total = X_total.drop(["speech"], axis=1)
    X_total["index"] = 0
    X_total["path"] = ""
    X_total["actor"] = pd.to_numeric(X_total["video_name"].str.replace(".csv", "").str.split('-').str[-1])
    X_total["emotion"] = pd.to_numeric(X_total["video_name"].str.split('-').str[2])
    X_total["emotion"]-=1
    return X_total



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-AUs', '--AUs_dir', type=str, required=True,
                        help='Path with the embeddings to train/test the models')
    parser.add_argument('-out', '--out_dir', type=str, help='Path to save the posteriors of the train and test sets after trainig each model',
                        default='')
    parser.add_argument('-m', '--model_number', type=int, required=True,
                        help='1-SVC / 2- Logistic Regression / 3- ridgeClassifier /4-perceptron / 5-NuSVC / 6-LinearSVC / 7-knn / 8-NearestCentroid / 9- DecrissionTree / 10- RandomForest / 11 - MLP')
    parser.add_argument('-modelParam', '--param', type=str, required=True,
                        help='Parameter of the model: C for SVC / C Logistic Regression / alpha in ridgeClassifier / alpha in perceptron / nu in NuSVC / C in LinearSVC / k in knn / None in NearestCentroid / min_samples_split in DecrissionTree / n_estimators in RandomForest / hidden_layer_sizes in MLP',
                        default=2)
    parser.add_argument('-norm', '--type_of_norm', type=int, required=True,
                        help='0-MinMax Norm / 1-Standard Norm / 2- No apply normalization [default: 2]', default=2)

    args = parser.parse_args()

    seed = 2020
    avg_embs_path = os.path.join(args.AUs_dir.rsplit("/", 1)[0], "df_average_AUs_total.csv")
    if(eval(args.out_dir)==""):
        get_embs = False
    else:
        get_embs = True
        os.makedirs(args.out_dir, exist_ok=True)

    # Get average:
    X_total = process_AUs_avg(args.AUs_dir, avg_embs_path)

    avg_acc = 0
    for fold in range(5):
        print("Processing fold: ", str(fold))
        train_df, test_df = generate_train_test(fold, X_total)

        X_train, y_train, _ = clean_df(train_df)
        X_test, y_test, _ = clean_df(test_df)

        if (int(args.type_of_norm) in [0, 1]):
            if (args.type_of_norm == 1):
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        classifier = get_classifier(args.model_number, args.param, seed=2020)
        classifier.fit(X_train, y_train)

        if(get_embs):
            train_path = os.path.join(args.out_dir, "train_fold"+str(fold)+".csv")
            extract_posteriors(classifier, X_train, train_df, train_path)
            test_path = os.path.join(args.out_dir, "test_fold" + str(fold) + ".csv")
            extract_posteriors(classifier, X_test, test_df, test_path)


        predictions = classifier.predict(X_test)
        accuracy = np.mean((y_test == predictions).astype(np.float)) * 100.
        avg_acc+=accuracy
        print(f"Accuracy = {accuracy:.3f}")
        print("------------")
    print("FINAL TEST ACCURACY: ", str(avg_acc/5))
