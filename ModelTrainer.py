from sklearn.model_selection import train_test_split

# Class for training and evaluating the model
class ModelTrainer:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self, batch_size, epochs=100):
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=2)
        return self.history

    def evaluate(self):
        train_loss, train_accuracy = self.model.evaluate(self.X_train, self.y_train)
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        return train_accuracy, test_accuracy
