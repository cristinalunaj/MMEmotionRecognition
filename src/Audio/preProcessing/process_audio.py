"""
    Main script to extract the audios at 16kHz and mono-channel from the videos

	author: Cristina Luna.
	date: 03/2022


	Usage:
		e.g.
		 python3 MMEmotionRecognition/src/Audio/FineTuningWav2Vec/process_audio.py
		 --videos_dir <RAVDESS_dir>/videos
		 --out_dir <RAVDESS_dir>/audios_16kHz

	Options:
        --videos_dir: Path to the directory with the videos of the dataset
		--out_dir: Path to save the audios generated at 16kHz and single channel (mono)
"""

import argparse
import os


def convert_audio216k(in_path_video, out_path_audio):
    if(os.path.exists(out_path_audio)):
        os.remove(out_path_audio)
    exit_code = os.system("ffmpeg -i " + in_path_video + " -ar 16k -ac 1 " + out_path_audio)

def extract_original_audio(in_path_video, out_path_audio):
    exit_code = os.system("ffmpeg -i "+in_path_video+" "+out_path_audio)




if __name__ == '__main__':
    # Read input parameters
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-videos', '--videos_dir', type=str, required=True,
                        help='Path with the videos to extract the audios')
    parser.add_argument('-out', '--out_dir', type=str, help='Path to save the processed audios at 16kHzs and mono channel',
                        default='./')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for video_i in os.listdir(args.videos_dir):
        path_video = os.path.join(args.videos_dir, video_i)
        path_out_audio = os.path.join(args.out_dir, video_i.rsplit(".")[0]+".wav")
        print("Processing ", path_video)
        convert_audio216k(path_video, path_out_audio)
