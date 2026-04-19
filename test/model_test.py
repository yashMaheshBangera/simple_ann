from model.custom_model_framework import custom_forward_pass, custom_model_weights_biases,train_step
import tensorflow as tf

def test_custom_model(X,y):
    # Final check on test data
    test_preds = custom_forward_pass(X)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_preds, axis=1), 
                                              tf.argmax(y, axis=1)), tf.float32))
    print(f"Final Test Accuracy: {accuracy.numpy() * 100:.2f}%")
    return accuracy.numpy() * 100