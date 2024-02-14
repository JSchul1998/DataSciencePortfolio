import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
import seaborn as sns
np.random.seed(0)

# Data Import from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Visualize the Examples
classes = 10
f, ax = plt.subplots(1, classes, figsize=(20,20))
for i in range(0, classes):
    # Visualize only first example (out of 60000) for MNIST digits 
    sample = x_train[y_train == i][0]
    ax[i].imshow(sample, cmap='viridis')
    ax[i].set_title(f'Label: {i}')
plt.show()

# One-Hot-Encode the label (y) data
y_train = keras.utils.to_categorical(y_train,classes)
y_test = keras.utils.to_categorical(y_test,classes)
# Ensure encoding is correct
for i in range(10):
    print(y_train[i])

# Data Normalization 
x_train = x_train/255.0
x_test = x_test/255.0
# Reshape x data from 2D array to flattened 1D
print('Pre-Flattened Shape:', x_train.shape)
x_train = x_train.reshape(x_train.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0],-1)
print('Post-Flattened Shape:', x_train.shape)


# Create fully connected neural network
model = Sequential()
model.add(Dense(units=128,input_shape=(x_test.shape[1],),activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0,25))
model.add(Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# Now train on the MNIST dataset
batch_size = 512
epochs = 10
model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs)

# Evaluate model performance and predict digits
test_loss, test_acc = model.evaluate(x_test,y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred,axis=1)
# This prints the probabilities for each digit
print(y_pred)
# This states the digit with the highest probability
print(y_pred_classes)

# Now check how well the model is performing on a single example
random_index = np.random.choice(len(x_test))
x_sample = x_test[random_index]
y_true = np.argmax(y_test,axis=1)
y_sample_true = y_true[random_index]
y_sample_pred_class = y_pred_classes[random_index]

plt.title(f'Predicted: {y_sample_pred_class}, True: {y_sample_true}')
plt.imshow(x_sample.reshape(28,28),cmap='viridis')
plt.show()


# Now, create a confusion matrix to see how well the model is performing
# across the digits

confusion = confusion_matrix(y_true,y_pred_classes)
fig,ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(confusion,annot=True,fmt='d',ax=ax,cmap='Blues')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.show()

