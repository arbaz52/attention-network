
from keras.layers import Layer
from keras import backend as K

tf = K.tf




class AttentionLayer(Layer):
  def __init__(self, n_cells):
    super(AttentionLayer, self).__init__()
    self.n_cells = n_cells
    
    #weights
    self.w1 = self.add_weight(name = "weight_context",
                             shape = (n_cells, 1),
                             trainable = True,
                             initializer = tf.random_uniform_initializer())
    self.w2 = self.add_weight(name = "weight_hidden",
                             shape = (n_cells, 1),
                             trainable = True,
                             initializer = tf.random_uniform_initializer())
    
  def call(self, inputs):
    hiddens, context = inputs, inputs[-1]
    
    hw = tf.matmul(hiddens, self.w2)
    cw = tf.matmul(context, self.w1)
    m = tf.nn.tanh(hw+cw)
    
    s = tf.nn.softmax(m, axis = 1) #please do not remove axis = 1, check documentation for this function
    e = s*hiddens
    
    z = tf.reduce_sum(e, axis = 1)
    z = tf.expand_dims(z, axis = -1)
    print(s.shape, z.shape)
    
    
    return [z, s]
  
  def compute_output_shape(self, input_shape):
    return [(None, self.n_cells, 1), (None, self.n_cells, 1)]