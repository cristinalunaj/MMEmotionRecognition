import os.path, os, sys
import argparse
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-AUs', '--AUs_dir', type=str, required=True,
                        help='Path with the AUs')
    parser.add_argument('-out', '--out_dir', type=str,
                        help='Path to save the plots of the AUs',
                        default='./')
    args = parser.parse_args()


    plt.rcParams['figure.figsize'] = (10, 5)
    emot_dict = {1:"Neutral",
                 2:"Calm",
                 3:"happy",
                 4:"sad",
                 5:"angry",
                 6:"fear",
                 7:"disgust",
                 8:"surprise"}
    plot_presence = True

    for au_file in os.listdir(args.AUs_dir):
        splitted_video_name = au_file.split("-")
        emotion = int(splitted_video_name[2]) #- 1  # See dict
        actor = splitted_video_name[-1].split(".")[0]
        emotional_intensity = int(splitted_video_name[3])  # 01 = normal, 02 = strong
        path_file = os.path.join(args.AUs_dir, au_file)
        df_AUs = pd.read_csv(path_file, sep=";", header=0)
        presenceAU_df = df_AUs[df_AUs.columns[17::]]
        intensity_AU_df = df_AUs[df_AUs.columns[0:17]]


        #Where are the most intense regions -> clear AUs
        x = range(0, len(intensity_AU_df))
        fig, ax = plt.subplots()  # figsize=(13,10)
        if(plot_presence):
            extra_name = "_presence"
            presenceAU_df.plot(colormap='tab20', grid=True)
            plt.title(au_file+" "+emot_dict[emotion]+" - "+str(emotional_intensity))
            plt.yticks(np.arange(0, 1.5, 0.5))
        else:
            extra_name = "_intensity"
            intensity_AU_df.plot(colormap='tab20',  grid=True)
            plt.title(au_file + " " + emot_dict[emotion] + " - " + str(emotional_intensity))
            plt.yticks(np.arange(0, 5.5, 0.5))
        #plt.show()

        os.makedirs(os.path.join(args.out_dir, "EMOTIONS",emot_dict[emotion]), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir,"ACTOR", actor), exist_ok=True)

        plt.savefig(os.path.join(args.out_dir, "EMOTIONS",emot_dict[emotion], au_file.replace(".csv", extra_name+".png")))
        plt.savefig(os.path.join(args.out_dir,"ACTOR", actor,au_file.replace(".csv", extra_name+".png")))

        plt.close('all')



