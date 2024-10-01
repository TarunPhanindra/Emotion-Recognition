import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from AudioDataLoader import AudioDataLoader
from CNNLSTMModel import CNNLSTMModel
from ModelTrainer import ModelTrainer


# Main execution part
if __name__ == '__main__':
    # Load and process the dataset
    Ravdess = r'/Users/tarun/Documents/Speech Emotion Recognition/VSCode/RAVDESS'
    data_loader = AudioDataLoader(Ravdess)
    mfccs, fileEmotion, ravdess_df = data_loader.load_audio_files_and_extract_mfcc()
    data_loader.replace_emotion_labels()
    data_loader.plot_emotion_distribution()

    # One-hot encoding of emotion labels
    encoder = OneHotEncoder()
    encoded_labels = encoder.fit_transform(np.array(fileEmotion).reshape(-1, 1)).toarray()
    num_classes = encoded_labels.shape[1]

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(mfccs, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)
    X_train = X_train.reshape(X_train.shape[0], 13, 228, 1)
    X_test = X_test.reshape(X_test.shape[0], 13, 228, 1)

    # Create, compile, and summarize the model
    cnn_lstm_model = CNNLSTMModel(input_shape=(13, 228, 1), num_classes=num_classes)
    cnn_lstm_model.compile_model(learning_rate=0.001)
    cnn_lstm_model.summary()

    # Train and evaluate the model
    trainer = ModelTrainer(cnn_lstm_model.model, X_train, y_train, X_test, y_test)
    trainer.train(batch_size=32, epochs=100)
    trainer.evaluate()
