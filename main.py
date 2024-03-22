import numpy as np

from DeepLearningAss1 import MLP

def main():
    
    train_data = np.load("./dataset/train_data.npy")
    train_label = np.load("./dataset/train_label.npy")
    test_data = np.load("./dataset/test_data.npy")
    test_label = np.load("./dataset/test_label.npy")

    nn = MLP(
        layers=[128, 80, 40, 10],
        activation=['relu', 'relu', 'relu', 'relu'],
    )
    nn.fit(
        X=train_data,
        y=train_label,
        learning_rate=0.001,
        epochs=100,
    )

    # output = nn.predict()


if (__name__ == "__main__"):
    main()
