import numpy as np

from DeepLearningAss1 import MLP

from sklearn.metrics import accuracy_score

DATASET_PATH="./dataset/"

def main():
    
    train_data = np.load(f"./{DATASET_PATH}/train_data.npy")
    train_label = np.load(f"./{DATASET_PATH}/train_label.npy")

    test_data = np.load(f"./{DATASET_PATH}/test_data.npy")
    test_label = np.load(f"./{DATASET_PATH}/test_label.npy")

    flattened_train_data = train_data.flatten()
    flattened_test_data = test_data.flatten()

    normalized_train_data = np.interp(flattened_train_data, (np.min(flattened_train_data), np.max(flattened_train_data)), (-1, 1))
    normalized_test_data = np.interp(flattened_test_data, (np.min(flattened_train_data), np.max(flattened_train_data)), (-1, 1))

    train_data = normalized_train_data.reshape(train_data.shape)
    test_data = normalized_test_data.reshape(test_data.shape)

    # To use cross entropy
    nn = MLP(
        layers=[128, 80, 40, 10],
        activation=['relu', 'relu', 'relu', None],
    )
    nn.fit(
        X=train_data,
        y=train_label,
        learning_rate=0.001,
        epochs=5,
        # momentum=0.9,
        weight_decay=0.01, # TODO: momentum and weight decay work independently, can they work together?
        loss_fn='CCE',
        optimizer='SGD',
    )

    output = nn.predict(test_data)
    print("Accuracy: {:.1f}%".format(accuracy_score(test_label, output) * 100))


if (__name__ == "__main__"):
    main()
