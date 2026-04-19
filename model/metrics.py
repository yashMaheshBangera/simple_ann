import tensorflow as tf

def custom_activation_reLU(x):
    return tf.maximum(0,x)

def custom_activation_softmax(x):
    exp_x = tf.exp(x - tf.reduce_max(x, axis=1, keepdims=True))
    return exp_x / tf.reduce_sum(exp_x, axis=1, keepdims=True)

def custom_cross_entropy(y_true,y_pred):
    # Clip predictions to prevent log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
    # Calculate cross-entropy loss
    loss = -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))
    return loss