from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation

from keras.datasets import mnist

import numpy as np
from keras.utils import np_utils



#loading and preprocessing the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0


y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


#creating the model

layer_input = Input(x_train.shape[1:])

layer_hidden = CuDNNLSTM(128, return_sequences = True)(layer_input)
layer_hidden = Activation('relu')(layer_hidden)
layer_hidden = Dropout(0.2)(layer_hidden)

layer_hidden, attention_weights = AttentionLayer(128)(layer_hidden)

layer_hidden = CuDNNLSTM(128)(layer_hidden)
layer_hidden = Activation('relu')(layer_hidden)
layer_hidden = Dropout(0.2)(layer_hidden)

layer_hidden = Dense(64, activation='relu')(layer_hidden)
layer_hidden = Dropout(0.2)(layer_hidden)

layer_output = Dense(10, activation='softmax')(layer_hidden)

model = Model(layer_input, layer_output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


#training the model

m_loss = model.fit(x_train, y_train, epochs = 3, batch_size = 32, validation_data=(x_test, y_test))


#testing the results


#creating a model to get the attention weights for the given test values.
a_model = Model(layer_input, attention_weights)


pred = model.predict(x_test[:10, :, :])
at = a_model.predict(x_test[:10, :, :])
print(np.argmax(pred, axis = 1))
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+ 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(at[i].reshape(28, 1))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()