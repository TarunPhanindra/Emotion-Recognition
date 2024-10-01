from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Bidirectional, Flatten, LSTM, Dense, Dropout, Reshape, Attention, Layer
from keras.optimizers import Adam

#Class for creating the CNN-LSTM model
class CNNLSTMModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(0.3))

        model.add(Reshape((-1, 64)))

        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def compile_model(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        return self.model.summary()