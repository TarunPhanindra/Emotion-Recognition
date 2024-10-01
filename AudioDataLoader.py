import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Class for loading audio files and extracting MFCC features
class AudioDataLoader:
    def __init__(self, dataset_path, n_mfcc=13, max_len=228):
        self.dataset_path = dataset_path
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.file_emotion = []
        self.file_path = []
        self.mfccs = []
        self.ravdess_df = None

    def load_audio_files_and_extract_mfcc(self):
        main_folder = os.path.join(self.dataset_path, 'audio_speech_actors_01-24')
        subfolders = os.listdir(main_folder)

        for subfolder in subfolders:
            subfolder_path = os.path.join(main_folder, subfolder)

            if os.path.isdir(subfolder_path):
                files = os.listdir(subfolder_path)

                for file in files:
                    if file.endswith('.wav'):
                        part = file.split('.')[0].split('-')
                        emotion = int(part[2])
                        self.file_emotion.append(emotion)

                        file_path = os.path.join(subfolder_path, file)
                        self.file_path.append(file_path)

                        audio, sr = librosa.load(file_path, sr=22050)
                        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
                        mfcc = np.pad(mfcc, ((0, 0), (0, self.max_len - mfcc.shape[1])), mode='constant')
                        self.mfccs.append(mfcc)

        self.mfccs = np.array(self.mfccs)
        self.file_emotion = np.array(self.file_emotion)
        self.file_path = np.array(self.file_path)

        emotion_df = pd.DataFrame(self.file_emotion, columns=['Emotions'])
        path_df = pd.DataFrame(self.file_path, columns=['Path'])
        mfccs_df = pd.DataFrame(self.mfccs.reshape(self.mfccs.shape[0], -1))
        self.ravdess_df = pd.concat([emotion_df, path_df, mfccs_df], axis=1)
        return self.mfccs, self.file_emotion, self.ravdess_df

    def replace_emotion_labels(self):
        self.ravdess_df.Emotions.replace(
            {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, 
            inplace=True
        )

    def plot_emotion_distribution(self):
        plt.title('Count of Emotions', size=16)
        sns.countplot(x='Emotions', data=self.ravdess_df)
        plt.ylabel('Count', size=12)
        plt.xlabel('Emotions', size=12)
        sns.despine(top=True, right=True, left=False, bottom=False)
        plt.show()
