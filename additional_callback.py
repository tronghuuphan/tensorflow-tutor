import tensorflow as tf

# Load dataset
from sklearn.datasets import load_diabetes
diabetes_dataset = load_diabetes()

# Save the input and target
from sklearn.model_selection import train_test_split
data = diabetes_dataset['data']
targets = diabetes_dataset['target']

# Split data
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
    ])

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

# Define the learning rate schedule function
def lr_function(epoch, lr):
    if epoch % 2 ==0:
        return lr
    else:
        return lr + epoch/1000

# Train the model 
history = model.fit(train_data, train_targets, epochs=10,
        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_function, verbose=1), tf.keras.callbacks.CSVLogger('result.csv')], verbose=False)

