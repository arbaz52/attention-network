# attention-network
**Version 0.1**
Contains a custom Keras Layer "Attention Layer".
Outputs: 
- Focused Inputs
- Attention Weights (Softmax Values)

***
## Contributers
- Arbaz Ajaz <arbaz5256@gmail.com>
- Kanwal Shariq <kanwalshariq@gmail.com>

***

## Usage

Simply pass in the number of cells in the previous LSTM/RNN Layer in the constructor.
```Python
  layer_hidden = CuDNNLSTM(128, return_sequences = True)(layer_input) #Previous LSTM Layer
  layer_hidden = Activation('relu')(layer_hidden)
  layer_hidden = Dropout(0.2)(layer_hidden)

  layer_hidden, attention_weights = AttentionLayer(128)(layer_hidden) #Our Custom Layer
```
The above lines of code are from a problem that has also been provided, It tells which number is written/in the picture.

