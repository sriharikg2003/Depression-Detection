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
# FUNCTION TO GENERATE MODULATION SPECTROGRAM IMAGE WITH ERROR HANDLING
def specImage(filename, win_size_sec=0.04):
    try:
        fs, x = wavfile.read(filename)
        x_name = ['speech']
        x = x / np.max(x)

        X_data = modSpec(x, fs, win_size_sec)

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
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None  # Return None to indicate an error

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

# Modify the generate_df function to handle errors
def generate_df(EATD, win_size_sec=0.04):
    EATD_SPEC_TRAIN = []
    EATD_SPEC_TEST = []

    for key in EATD.keys():
        column = key
        code_value = EATD[column]['Code']

        for i in tqdm(EATD[column]['Storage']):
            img = specImage(i, win_size_sec)
            if img is not None:  # Check if an image was successfully generated
                if column.startswith('TRAIN'):
                    EATD_SPEC_TRAIN.append({"Image": img, "Code": code_value})
                elif column.startswith('TEST'):
                    EATD_SPEC_TEST.append({"Image": img, "Code": code_value})

    return EATD_SPEC_TRAIN, EATD_SPEC_TEST

# ...


import pickle

# load_path = ""
save_path = "Mod_Spec_Images/"

# Assuming you have already loaded your data into eatd_df_train and eatd_df_test

# Saving dataframes

# Modify the loop to handle different window sizes
win_sizes = [ 0.4]  # Adjust the window sizes as needed
name_of_file = [str(x) for x in win_sizes]

for i in range(len(win_sizes)):
    print(f"Saving as 'Above_3_seconds_0_{name_of_file[i].split('.')[-1]}.pkl'")
    eatd_df_train_win_sizes, eatd_df_test_win_sizes = generate_df(EATD, win_size_sec=win_sizes[i])
    eatd_df_train_win_sizes = pd.DataFrame(eatd_df_train_win_sizes)
    eatd_df_test_win_sizes = pd.DataFrame(eatd_df_test_win_sizes)

    with open(save_path + f'Above_3_seconds_train_0_{name_of_file[i].split(".")[-1]}.pkl', 'wb') as f:
        pickle.dump(eatd_df_train_win_sizes, f)

    with open(save_path + f'Above_3_seconds_test_0_{name_of_file[i].split(".")[-1]}.pkl', 'wb') as f:
        pickle.dump(eatd_df_test_win_sizes, f)

    print(f"Saved 'Above_3_seconds_0_{name_of_file[i].split('.')[-1]}.pkl'")
    
print("Done")