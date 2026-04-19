import tensorflow as tf
from tensorflow.keras import layers
from model.metrics import custom_activation_reLU, custom_activation_softmax, custom_cross_entropy

def keras_model():
    model = tf.keras.Sequential([
        # 28x28 input images 
        layers.Input(shape=(28,28)),
        # Flatten to 784 features
        layers.Flatten(),
        # Hidden layer with 128 neurons and ReLU activation
        layers.Dense(128, activation='relu'),
        # Hidden layer with 128 neurons and ReLU activation
        layers.Dense(128,activation='relu'),
        # Output layer with 10 neurons ( for 10 classes ) and softmax activation
        layers.Dense(10, activation='softmax')
    ])
    return model

class custom_model():
    def __init__(self, default_learning_rate=None):
        # Input layer to Hidden layer (28*28 input features to 128 neurons)
        self.W1 = tf.Variable(tf.random.normal([28 * 28, 128], stddev=0.1), name='weight1')
        self.b1 = tf.Variable(tf.zeros([128]), name='bias1')

        # Hidden layer to Hidden layer (128 neurons to 128 neurons)
        self.W2 = tf.Variable(tf.random.normal([128, 128], stddev=0.1), name='weight2')
        self.b2 = tf.Variable(tf.zeros([128]), name='bias2')

        # Hidden layer to output (1 prediction)
        self.W3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1), name='weight3')
        self.b3 = tf.Variable(tf.zeros([10]), name='bias3')

    def custom_forward_pass(self, X):
        # 1. Hidden Layer
        # Weighted sum + bias
        z1 = tf.add(tf.matmul(X, self.W1), self.b1)
        # ReLU: max(0, z1) - filters out negative values
        a1 = custom_activation_reLU(z1) 

        # 2. Hidden layer to Hidden layer
        # Weighted sum + bias
        z2 = tf.add(tf.matmul(a1, self.W2), self.b2)
        # ReLU: max(0, z2)
        a2 = custom_activation_reLU(z2)

        # 3. Output Layer
        # Weighted sum + bias
        z3 = tf.add(tf.matmul(a2,self.W3),self.b3)
        # Softmax: Converts scores into probabilities (sum to 1.0)
        output = custom_activation_softmax(z3)

        return output

    def train_step(self, X, y, learning_rate=None):
        lr = learning_rate if learning_rate is not None else self.default_learning_rate
        if lr is None:
            raise ValueError("Learning rate must be provided either as an argument or set as default_learning_rate during initialization.")
        with tf.GradientTape() as tape:
            # 1. Forward Pass
            predictions = self.custom_forward_pass(X)

            # 2. Calculate Loss
            loss = custom_cross_entropy(y, predictions)
        
        # 3. Get Gradients: "How should I change W and b to make loss smaller?"
        variables = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        gradients = tape.gradient(loss, variables)
        
        # 4. Optimizer: Update weights using Gradient Descent
        # weight = weight - (learning_rate * gradient)
        for var, grad in zip(variables, gradients):
            var.assign_sub(lr * grad)
            
        return loss,predictions
    
    def predict(self,X):
        return self.custom_forward_pass(X)
    
    def evaluate(self, X,y):
        final_predictions = self.custom_forward_pass(X)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final_predictions, axis=1), 
                                              tf.argmax(y, axis=1)), tf.float32))
        return accuracy.numpy() * 100
        