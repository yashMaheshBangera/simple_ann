import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from model.custom_model_framework import custom_model
import argparse

def model_train(
        batch_size,
        epochs,
        lr
):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Flatten 28x28 to 784 and normalize to [0, 1]
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    # Convert integer labels (0-9) to One-Hot vectors
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    batch_size = batch_size
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)

    epochs = epochs
    learning_rate = lr

    model = custom_model()

    for epoch in range(epochs):
        epoch_loss = 0
        for step, (x_batch, y_batch) in enumerate(train_ds):
            # Call the train_step we built earlier
            loss,predictions = model.train_step(x_batch, y_batch, learning_rate)
            epoch_loss += loss
            
        print(f"Epoch {epoch + 1}: Avg Loss = {epoch_loss / (step + 1):.4f}")
    print(f"Training process completed. Model was trained for {epochs} epochs with batch size {batch_size} and learning rate {lr}.")
    final_test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy : {final_test_accuracy:.2f}%")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom ANN model on the MNIST dataset.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate for training")
    args = parser.parse_args()
    model_train(args.batch_size, args.epochs, args.learning_rate)