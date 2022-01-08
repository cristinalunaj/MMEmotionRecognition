import os.path, os, sys
import argparse
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

import pandas as pd
import numpy as np
from src.Audio.FeatureExtractionWav2Vec.FeatureTraining import get_classifier, extract_posteriors
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) # only difference

def remove_cols(df, cols2rm=[]):
    for col in cols2rm:
        try:
            df = df.drop(columns=[col])
        except KeyError:
            continue
    return df


def prepare_video_modality(X_total, fold):
    actors_per_fold = {
        0: [2, 5, 14, 15, 16],
        1: [3, 6, 7, 13, 18],
        2: [10, 11, 12, 19, 20],
        3: [8, 17, 21, 23, 24],
        4: [1, 4, 9, 22],
    }
    X_total["actor"] +=1

    test_df = X_total.loc[X_total['actor'].isin(actors_per_fold[fold])]
    train_df = X_total.loc[~X_total['actor'].isin(actors_per_fold[fold])]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df

def prepare_video_modality_biLSTMAtt(df):
    #Convert emotions:
    df["emotion"] = df["tag"].str.split("-").str[2].astype(int)
    df["emotion"] -= 1
    #Extend posteriors columns
    df[["posterios"+str(i) for i in range(8)]] = df["posteriors"].str.split("[").str[-1].str.split("]").str[0].str.split(",", expand=True)
    df[["posterios"+str(i) for i in range(8)]] = df[["posterios"+str(i) for i in range(8)]].apply(pd.to_numeric)
    df["tag"] = df["tag"].str.replace(".csv", "")
    #remove cols:
    df = remove_cols(df, cols2rm=[ "y", "y_hat", "posteriors", "attentions"])
    df = df.rename(columns = {"tag":"name"})
    return df



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-embsWav2vec', '--embs_dir_wav2vec', type=str, required=True,
                        help='Path with the embeddings of the fine-tuned Wav2Vec model')
    parser.add_argument('-embsBiLSTM', '--embs_dir_biLSTM', type=str, required=True,
                        help='Path with the embeddings of the bi-LSTM with attention mechanism trained with the AUs')
    parser.add_argument('-embsMLP', '--embs_dir_MLP', type=str, required=True,
                        help='Path with the embeddings of the MLP trained with the average of the AUs')
    parser.add_argument('-out', '--out_dir', type=str, help='Path to save the embeddings extracted from the model [default: Not save embs]',
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

    if (eval(args.out_dir) == ""):
        get_embs = False
    else:
        get_embs = True
        os.makedirs(args.out_dir, exist_ok=True)


    avg_acc = 0
    for fold in range(5):
        # AUDIO WAV2VEC MODEL
        X_train_audio = pd.read_csv(os.path.join(args.embs_dir_wav2vec, "fold" + str(fold), "posteriors_train.csv"), sep=";",
                                    header=0)
        X_test_audio = pd.read_csv(os.path.join(args.embs_dir_wav2vec, "fold" + str(fold), "posteriors_test.csv"), sep=";",
                                   header=0)

        X_train_audio = X_train_audio.sort_values(by=["name"])
        X_train_audio = X_train_audio.reset_index(drop=True)

        X_test_audio = X_test_audio.sort_values(by=["name"])
        X_test_audio = X_test_audio.reset_index(drop=True)

        # VIDEO SEQUENTIAL MODELS
        X_train_video = pd.read_csv(
            os.path.join(args.embs_dir_biLSTM, "RAVDESS_AUs-train--fold_" + str(fold + 1) + "_outof_5.csv"), sep="\t",
            header=0)
        X_test_video = pd.read_csv(
            os.path.join(args.embs_dir_biLSTM, "RAVDESS_AUs-val--fold_" + str(fold + 1) + "_outof_5.csv"), sep="\t",
            header=0)
        X_train_video = prepare_video_modality_biLSTMAtt(X_train_video)
        X_test_video = prepare_video_modality_biLSTMAtt(X_test_video)

        X_train_video = X_train_video.sort_values(by=["name"])
        X_train_video = X_train_video.reset_index(drop=True)

        X_test_video = X_test_video.sort_values(by=["name"])
        X_test_video = X_test_video.reset_index(drop=True)

        # VIDEO AVG - STATIC MODELS
        X_train_video_avg = pd.read_csv(os.path.join(args.embs_dir_MLP, "train_fold"+str(fold)+".csv"), sep=";", header=0)
        X_test_video_avg = pd.read_csv(os.path.join(args.embs_dir_MLP, "test_fold" + str(fold) + ".csv"), sep=";",
                                        header=0)
        X_train_video_avg = X_train_video_avg.sort_values(by=["video_name"])
        X_train_video_avg = X_train_video_avg.reset_index(drop=True)

        X_test_video_avg = X_test_video_avg.sort_values(by=["video_name"])
        X_test_video_avg = X_test_video_avg.reset_index(drop=True)

        # AUDIO AVG - STATIC MODELS
        X_train_audio_avg = pd.read_csv(os.path.join(args.embs_dir_MLP, "train_fold" + str(fold) + ".csv"), sep=";",
                                        header=0)
        X_test_audio_avg = pd.read_csv(os.path.join(args.embs_dir_MLP, "test_fold" + str(fold) + ".csv"), sep=";",
                                       header=0)
        X_train_audio_avg = X_train_audio_avg.sort_values(by=["video_name"])
        X_train_audio_avg = X_train_audio_avg.reset_index(drop=True)

        X_test_audio_avg = X_test_audio_avg.sort_values(by=["video_name"])
        X_test_audio_avg = X_test_audio_avg.reset_index(drop=True)


        #Remove audio cols:
        y_train = pd.DataFrame([])
        y_test = pd.DataFrame([])
        y_train["emotion"] = X_train_video["emotion"]
        y_test["emotion"] = X_test_video["emotion"]
        #Remove cols:
        X_train_audio = X_train_audio.rename(columns={"name":"NAME", "emotion":"EMOTION"}) #"actor":"ACTOR"
        X_test_audio = X_test_audio.rename(columns={"name": "NAME", "emotion": "EMOTION"})  # "actor":"ACTOR"

        #Combine data
        #
        X_train_MM = pd.concat([X_train_audio, X_train_video,X_train_video_avg[["embs"+str(i) for i in range(8)]]], axis=1) #X_train_audio_avg[["embs"+str(i) for i in range(8)]]
        #
        X_test_MM = pd.concat([X_test_audio, X_test_video, X_test_video_avg[["embs"+str(i) for i in range(8)]]], axis=1) #X_test_audio_avg[["embs"+str(i) for i in range(8)]]

        #Remove columns
        X_train_MM = remove_cols(X_train_MM, cols2rm=["NAME", "EMOTION", "ACTOR",
                                                      "name", "emotion", "actor", "fold", "weigths"])

        X_test_MM = remove_cols(X_test_MM, cols2rm=["NAME", "EMOTION", "ACTOR",
                                                      "name", "emotion", "actor", "fold", "weigths"])

        # randomize data
        X_train_MM = shuffle(X_train_MM,random_state=seed)
        y_train = shuffle(y_train, random_state=seed)
        X_train_MM = X_train_MM.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        if (args.type_of_norm in [0, 1]):
            if (args.type_of_norm == 1):
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                scaler = StandardScaler()
            X_train_MM = scaler.fit_transform(X_train_MM)
            X_test_MM = scaler.transform(X_test_MM)
        # Train models
        classifier = get_classifier(args.model_number, args.param, seed=seed)
        classifier.fit(X_train_MM, y_train)

        if (get_embs):
            train_path = os.path.join(args.out_dir, "train_fold" + str(fold) + ".csv")
            extract_posteriors(classifier, X_train_MM, X_test_video_avg, train_path)
            test_path = os.path.join(args.out_dir, "test_fold" + str(fold) + ".csv")
            extract_posteriors(classifier, X_test_MM, X_test_video_avg, test_path)

        predictions = classifier.predict(X_test_MM)
        accuracy = np.mean((y_test["emotion"] == predictions).astype(np.float)) * 100.
        avg_acc += accuracy
        print(f"Accuracy = {accuracy:.3f}")
        print("------------")
    print("FINAL TEST ACCURACY: ", str(avg_acc / 5))




