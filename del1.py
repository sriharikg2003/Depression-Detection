print("IMPORTED")

import sys
from am_analysis import am_analysis as ama


import skimage.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
print("IMPORTED")

from scipy.io import wavfile
import glob
from IPython.display import Audio
from tqdm import tqdm
import pandas as pd
print("IMPORTED")


# FUNCTIONS FOR MODULATION SPECTROGRAM
def modSpec(x, fs,win_size_sec=0.04):
    # win_size_sec = 0.04  # window length for the STFFT (seconds)
    win_shft_sec = 0.01  # shift between consecutive windows (seconds)

    stft_modulation_spectrogram = ama.strfft_modulation_spectrogram(
        x,
        fs,
        win_size=round(win_size_sec * fs),
        win_shift=round(win_shft_sec * fs))

    return stft_modulation_spectrogram

def specImage(filename,win_size_sec=0.04):
    fs, x = wavfile.read(filename)
    x_name = ['speech']
    x = x / np.max(x)
    # 1s segment to analyze
    # x = x[int(fs*1.6) : int(fs*3.6)]

    X_data = modSpec(x, fs,win_size_sec)

    ama.plot_modulation_spectrogram_data(X_data,
                                         0,
                                         modf_range=np.array([0, 20]),
                                         c_range=np.array([-90, -50]))

    # Get the current figure and convert it to a 3D array
    fig = plt.gcf()
    fig.canvas.draw()
    plot_data_rgba = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()  # Close the plot to free up resources

    # Remove the alpha channel to get a 3D array
    plot_data_rgb = plot_data_rgba[:, :, :3]

    return plot_data_rgb
def ssimFromAudio(filepath1, filepath2,win_size=11):
    img1 = specImage(filepath1)
    img2 = specImage(filepath2)
    ssim_score = metrics.structural_similarity(img1, img2, win_size=win_size, channel_axis=2)
    return ssim_score

def ssimFromImage(img1, img2,win_size=11):

    ssim_score = metrics.structural_similarity(img1, img2, win_size=win_size, channel_axis=2)
    return ssim_score

def playAudio(path):
    return Audio(path)
# filepath1 = "TrainPHQ8/TrainPHQ8/train_depreesed/F_D_321(PHQ8-20)/321_21.wav"


EATD = {
    "TRAIN_D": {"src": "Above_3_seconds/Train/Train_D/", "Storage" : [], "Code" : 1},
    "TRAIN_ND": {"src":  "Above_3_seconds/Train/Train_ND/", "Storage" :[] ,"Code" : 0},
    "TEST_D": {"src":"Above_3_seconds/Test/Test_D/", "Storage": [],"Code" : 1},
    "TEST_ND": {"src": "Above_3_seconds/Test/Test_ND/", "Storage": [],"Code" : 0}
}
for key in EATD.keys():
    EATD[key]["Storage"] =  glob.glob(EATD[key]["src"] + "*")


def generate_df (EATD,win_size_sec=0.04):
    EATD_SPEC_TRAIN = []

    column = 'TRAIN_D'
    code_value = EATD[column]['Code']  # Get the code value outside the loop

    for i in tqdm(EATD[column]['Storage']):
        try:
            EATD_SPEC_TRAIN.append({"Image": specImage(i,win_size_sec), "Code": code_value})
        except:
            print(f"Error in {i}")

    column = 'TRAIN_ND'
    code_value = EATD[column]['Code']  # Get the code value outside the loop

    for i in tqdm(EATD[column]['Storage']):
        try:
            EATD_SPEC_TRAIN.append({"Image": specImage(i,win_size_sec), "Code": code_value})
        except:
            print(f"Error in {i}")





    EATD_SPEC_TEST= []

    column = 'TEST_D'
    code_value = EATD[column]['Code']  # Get the code value outside the loop

    for i in tqdm(EATD[column]['Storage']):
        try:
            EATD_SPEC_TEST.append({"Image": specImage(i,win_size_sec), "Code": code_value})
        except:
            print(f"Error in {i}")

    column = 'TEST_ND'
    code_value = EATD[column]['Code']  # Get the code value outside the loop

    for i in tqdm(EATD[column]['Storage']):
        try:
            EATD_SPEC_TEST.append({"Image": specImage(i,win_size_sec), "Code": code_value})
        except:
            print(f"Error in {i}")
    return EATD_SPEC_TRAIN,EATD_SPEC_TEST


import pickle

# load_path = ""
save_path = "Mod_Spec_Images/"

# Assuming you have already loaded your data into eatd_df_train and eatd_df_test

# Saving dataframes

win_sizes = [0.025,0.4,0.8]
name_of_file = [str(x) for x in win_sizes]
for i in range(3):
    print("Saving as  "+  'Above_3_seconds_0_'+name_of_file[i].split(".")[-1]+'.pkl')
    print("Saving as " + 'Above_3_seconds_0_'+name_of_file[i].split(".")[-1]+'.pkl')
    eatd_df_train_win_sizes ,eatd_df_test_win_sizes = generate_df (EATD,win_size_sec=win_sizes[i])
    eatd_df_train_win_sizes = pd.DataFrame(eatd_df_train_win_sizes)
    eatd_df_test_win_sizes = pd.DataFrame(eatd_df_test_win_sizes)
    with open(save_path + 'Above_3_seconds_train_0_'+name_of_file[i].split(".")[-1]+'.pkl', 'wb') as f:
        pickle.dump(eatd_df_train_win_sizes, f)

    with open(save_path + 'Above_3_seconds_test_0_'+name_of_file[i].split(".")[-1]+'.pkl', 'wb') as f:
        pickle.dump(eatd_df_test_win_sizes, f)
    print("Saved "+  'Above_3_seconds_0_'+name_of_file[i].split(".")[-1]+'.pkl')
    print("Saved" + 'Above_3_seconds_0_'+name_of_file[i].split(".")[-1]+'.pkl')
print("Done")