import tensorflow as tf

# Input size (e.g., 28x28 images) and hidden layer size (e.g., 128 neurons)
W1 = tf.Variable(tf.random.normal([28 * 28, 128], stddev=0.1), name='weight1')
b1 = tf.Variable(tf.zeros([128]), name='bias1')

# Hidden layer to Hidden layer (128 neurons to 128 neurons)
W2 = tf.Variable(tf.random.normal([128, 128], stddev=0.1), name='weight2')
b2 = tf.Variable(tf.zeros([128]), name='bias2')

# Hidden layer to output (1 prediction)
W3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1), name='weight3')
b3 = tf.Variable(tf.zeros([10]), name='bias3')

def relu(x):
    return tf.maximum(0, x)

def softmax(x):
    exp_x = tf.exp(x - tf.reduce_max(x, axis=1, keepdims=True))
    return exp_x / tf.reduce_sum(exp_x, axis=1, keepdims=True)

def forward_pass(X):
    # 1. Hidden Layer
    # Weighted sum + bias
    z1 = tf.add(tf.matmul(X, W1), b1)
    # ReLU: max(0, z1) - filters out negative values
    a1 = relu(z1) 

    # 2. Hidden layer to Hidden layer
    # Weighted sum + bias
    z2 = tf.add(tf.matmul(a1, W2), b2)
    # ReLU: max(0, z2)
    a2 = relu(z2)
    
    # 3. Output Layer
    # Weighted sum + bias
    z3 = tf.add(tf.matmul(a2, W3), b3)
    # Softmax: Converts scores into probabilities (sum to 1.0)
    output = softmax(z3)
    
    return output

def cross_entropy(y_true,y_pred):
    # Clip predictions to prevent log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
    # Calculate cross-entropy loss
    loss = -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))
    return loss

def train_step(X, y, learning_rate=0.01):
    with tf.GradientTape() as tape:
        # 1. Forward Pass
        predictions = forward_pass(X)
        
        # 2. Calculate Loss
        loss = cross_entropy(y, predictions)
    
    # 3. Get Gradients: "How should I change W and b to make loss smaller?"
    variables = [W1, b1, W2, b2, W3, b3]
    gradients = tape.gradient(loss, variables)
    
    # 4. Optimizer: Update weights using Gradient Descent
    # weight = weight - (learning_rate * gradient)
    for var, grad in zip(variables, gradients):
        var.assign_sub(learning_rate * grad)
        
    return loss

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten 28x28 to 784 and normalize to [0, 1]
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Convert integer labels (0-9) to One-Hot vectors
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

batch_size = 64
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)

epochs = 100
learning_rate = 0.05

for epoch in range(epochs):
    epoch_loss = 0
    for step, (x_batch, y_batch) in enumerate(train_ds):
        # Call the train_step we built earlier
        loss = train_step(x_batch, y_batch, learning_rate)
        epoch_loss += loss
        
    print(f"Epoch {epoch + 1}: Avg Loss = {epoch_loss / (step + 1):.4f}")

# Final check on test data
test_preds = forward_pass(x_test)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_preds, axis=1), 
                                          tf.argmax(y_test, axis=1)), tf.float32))
print(f"Final Test Accuracy: {accuracy.numpy() * 100:.2f}%")