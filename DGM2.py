# CLASS DEFINITIONS FOR NEURAL NETWORKS

#%% import needed packages
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#%% LSTM-like layer used in DGM (see Figure 5.3 and set of equations on p. 45) - modification of Keras layer class

class LSTMLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, trans1 = "tanh", trans2 = "tanh"):
        '''
        Args:
            input_dim (int):       dimensionality of input data
            output_dim (int):      number of outputs for LSTM layers
            trans1, trans2 (str):  activation functions used inside the layer; 
                                   one of: "tanh" (default), "relu" or "sigmoid"
        
        Returns: customized Keras layer object used as intermediate layers in DGM
        '''        
        
        # create an instance of a Layer object (call initialize function of superclass of LSTMLayer)
        super(LSTMLayer, self).__init__()
        
        # add properties for layer including activation functions used inside the layer  
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        if trans1 == "tanh":
            self.trans1 = tf.nn.tanh
        elif trans1 == "relu":
            self.trans1 = tf.nn.relu
        elif trans1 == "sigmoid":
            self.trans1 = tf.nn.sigmoid
        
        if trans2 == "tanh":
            self.trans2 = tf.nn.tanh
        elif trans2 == "relu":
            self.trans2 = tf.nn.relu
        elif trans2 == "sigmoid":
            self.trans2 = tf.nn.relu
        
        ### define LSTM layer parameters (use Xavier initialization)
        # u vectors (weighting vectors for inputs original inputs x)
        self.Uz = self.add_weight("Uz", shape=[self.input_dim, self.output_dim],
                                    initializer=tf.keras.initializers.glorot_normal)
        self.Ug = self.add_weight("Ug", shape=[self.input_dim, self.output_dim],
                                    initializer=tf.keras.initializers.glorot_normal)
        self.Ur = self.add_weight("Ur", shape=[self.input_dim, self.output_dim],
                                    initializer=tf.keras.initializers.glorot_normal)
        self.Uh = self.add_weight("Uh", shape=[self.input_dim, self.output_dim],
                                    initializer=tf.keras.initializers.glorot_normal)

        # w vectors (weighting vectors for output of previous layer)
        self.Wz = self.add_weight("Wz", shape=[self.output_dim, self.output_dim],
                                    initializer=tf.keras.initializers.glorot_normal)
        self.Wg = self.add_weight("Wg", shape=[self.output_dim, self.output_dim],
                                    initializer=tf.keras.initializers.glorot_normal)
        self.Wr = self.add_weight("Wr", shape=[self.output_dim, self.output_dim],
                                    initializer=tf.keras.initializers.glorot_normal)
        self.Wh = self.add_weight("Wh", shape=[self.output_dim, self.output_dim],
                                    initializer=tf.keras.initializers.glorot_normal)


        # bias vectors
        self.bz = self.add_weight("bz", shape=[1, self.output_dim])
        self.bg = self.add_weight("bg", shape=[1, self.output_dim])
        self.br = self.add_weight("br", shape=[1, self.output_dim])
        self.bh = self.add_weight("bh", shape=[1, self.output_dim])
    
    
    # main function to be called 
    def call(self, S, X):
        '''Compute output of a LSTMLayer for a given inputs S,X .    

        Args:            
            S: output of previous layer
            X: data input
        
        Returns: customized Keras layer object used as intermediate layers in DGM
        '''   
        
        # compute components of LSTM layer output (note H uses a separate activation function)
        Z = self.trans1(tf.add(tf.add(tf.matmul(X,self.Uz), tf.matmul(S, self.Wz)), self.bz))
        G = self.trans1(tf.add(tf.add(tf.matmul(X,self.Ug), tf.matmul(S, self.Wg)), self.bg))
        R = self.trans1(tf.add(tf.add(tf.matmul(X,self.Ur), tf.matmul(S, self.Wr)), self.br))
        
        H = self.trans2(tf.add(tf.add(tf.matmul(X,self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))
        
        # compute LSTM layer output
        S_new = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z,S))
        
        return S_new

#%% Fully connected (dense) layer - modification of Keras layer class
   
class DenseLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, transformation=None):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            transformation:  activation function used inside the layer; using
                             None is equivalent to the identity map 
        
        Returns: customized Keras (fully connected) layer object 
        '''        
        
        # create an instance of a Layer object (call initialize function of superclass of DenseLayer)
        super(DenseLayer,self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        ### define dense layer parameters (use Xavier initialization)
        # w vectors (weighting vectors for output of previous layer)
        self.W = self.add_weight("W", shape=[self.input_dim, self.output_dim],
                                   initializer=tf.keras.initializers.glorot_normal)
        
        # bias vectors
        self.b = self.add_weight("b", shape=[1, self.output_dim])
        
        if transformation:
            if transformation == "tanh":
                self.transformation = tf.tanh
            elif transformation == "relu":
                self.transformation = tf.nn.relu
        else:
            self.transformation = transformation
    
    
    # main function to be called 
    def call(self,X):
        '''Compute output of a dense layer for a given input X 

        Args:                        
            X: input to layer            
        '''
        
        # compute dense layer output
        S = tf.add(tf.matmul(X, self.W), self.b)
                
        if self.transformation:
            S = self.transformation(S)
        
        return S

#%% Neural network architecture used in DGM - modification of Keras Model class
    
class DGMNet(tf.keras.Model):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate LSTM layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''  
        
        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DGMNet,self).__init__()
        
        # define initial layer as fully connected 
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim, transformation = "tanh")
        
        # define intermediate LSTM layers
        self.n_layers = n_layers
        self.LSTMLayerList = []
                
        for _ in range(self.n_layers):
            self.LSTMLayerList.append(LSTMLayer(layer_width, input_dim))
        
        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(1, layer_width, transformation = final_trans)
    
    
    # main function to be called  
    def call(self,x):
        '''            
        Args:
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (x)                
        '''  
        
        # define input vector as time-space pairs
        X = x
        
        # call initial layer
        S = self.initial_layer.call(X)
        
        # call intermediate LSTM layers
        for i in range(self.n_layers):
            S = self.LSTMLayerList[i].call(S,X)
        
        # call final LSTM layers
        result = self.final_layer.call(S)
        
        return result
    
    
class DCGMNet(tf.keras.Model):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, X_low, X_high, layer_width, n_layers_FFNN, n_layers_RNN, input_dim, final_trans=None):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate LSTM layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''

        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DCGMNet, self).__init__()

        # define initial layer as fully connected
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.n_layers_FFNN = n_layers_FFNN
        self.n_layers_RNN = n_layers_RNN


        # define scaling layer: for standarization

        self.initial_layer_scale = tf.keras.layers.Lambda(lambda x: 2.0*(x - X_low)/(X_high - X_low) - 1.0)
        # define intermediate Dense layers

        self.DenseLayerList = []
        for _ in range(self.n_layers_FFNN):
            if _ == 0:
                self.DenseLayerList.append(DenseLayer(
                    layer_width, input_dim, transformation="tanh"))
            else :
                self.DenseLayerList.append(DenseLayer(
                    layer_width, layer_width, transformation="tanh"))
            
        # define intermediate LSTM layers
        self.LSTMLayerList = []

        for _ in range(self.n_layers_RNN):
            self.LSTMLayerList.append(LSTMLayer(layer_width, input_dim))

        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(
            1, layer_width, transformation=final_trans)

    # main function to be called

    def call(self, x):
        '''            
        Args:
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (x)                
        '''

        # define input vector as time-space pairs
        X = x

        # call initial layer scaling
        S = self.initial_layer_scale(X)
        
        for i in range(self.n_layers_FFNN):
            S = self.DenseLayerList[i].call(S)
            
        # call intermediate LSTM layers
        for i in range(self.n_layers_RNN):
            S = self.LSTMLayerList[i].call(S, X)

        # call final layers
        result = self.final_layer.call(S)

        return result



class DCGM2Net(tf.keras.Model):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, X_low, X_high, layer_width, n_layers_FFNN, n_layers_RNN, input_dim, activation_FFNN, final_trans=None):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate LSTM layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''

        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DCGM2Net, self).__init__()

        # define initial layer as fully connected
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.n_layers_FFNN = n_layers_FFNN
        self.n_layers_RNN = n_layers_RNN


        # define scaling layer: for standarization

        self.initial_layer_scale = tf.keras.layers.Lambda(lambda x: 2.0*(x - X_low)/(X_high - X_low) - 1.0)
        self.initial_layer = DenseLayer(layer_width, input_dim, transformation = "tanh")

        # define intermediate LSTM layers
        self.LSTMLayerList = []

        for _ in range(self.n_layers_RNN):
            self.LSTMLayerList.append(LSTMLayer(layer_width, input_dim))


        # define intermediate Dense layers

        self.DenseLayerList = []
        for _ in range(self.n_layers_FFNN):
            self.DenseLayerList.append(DenseLayer(
                layer_width, layer_width, transformation=activation_FFNN))
            

        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(
            1, layer_width, transformation=final_trans)

    # main function to be called

    def call(self, x):
        '''            
        Args:
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (x)                
        '''

        # define input vector 
        X = x

        # call initial layer scaling
        X = self.initial_layer_scale(X)
        S = self.initial_layer.call(X)
        
        # call intermediate LSTM layers
        for i in range(self.n_layers_RNN):
            S = self.LSTMLayerList[i].call(S, X) 
            
        Inter_S = S
            
        for i in range(self.n_layers_FFNN):
            S = self.DenseLayerList[i].call(S)
            
        # call final layers
        result = self.final_layer.call(S+Inter_S)

        return result



class DCGM3Net(tf.keras.Model):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, X_low, X_high, layer_width, n_layers_FFNN, n_layers_RNN, input_dim, activation_FFNN, final_trans=None):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate LSTM layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''

        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DCGM3Net, self).__init__()

        # define initial layer as fully connected
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.n_layers_FFNN = n_layers_FFNN
        self.n_layers_RNN = n_layers_RNN


        # define scaling layer: for standarization

        self.initial_layer_scale = tf.keras.layers.Lambda(lambda x: 2.0*(x - X_low)/(X_high - X_low) - 1.0)
        self.initial_layer = DenseLayer(layer_width, input_dim, transformation = "tanh")

        # define intermediate LSTM layers
        self.LSTMLayerList = []

        for _ in range(self.n_layers_RNN):
            self.LSTMLayerList.append(LSTMLayer(layer_width, input_dim))


        # define intermediate Dense layers

        self.DenseLayerList = []
        for _ in range(self.n_layers_FFNN):
            self.DenseLayerList.append(DenseLayer(
                layer_width, layer_width, transformation=activation_FFNN))
            

        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(
            1, layer_width, transformation=final_trans)

    # main function to be called

    def __call__(self, x):
        '''            
        Args:
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (x)                
        '''

        # define input vector 
        X = x

        # call initial layer scaling
        X = self.initial_layer_scale(X)
        S = self.initial_layer.call(X)
        
        # call intermediate LSTM layers
        for i in range(self.n_layers_RNN):
            S = self.LSTMLayerList[i].call(S, X) 
            
        Inter_S = S
            
        for i in range(self.n_layers_FFNN):
            S = self.DenseLayerList[i].call(S) + S
            
        # call final layers
        result = self.final_layer.call(S+Inter_S)

        return result



class DCGM4Net(tf.keras.Model):

    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, X_low, X_high, layer_width, n_layers_FFNN, n_layers_RNN, input_dim, output_dim, activation_FFNN, final_trans=None):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate LSTM layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''

        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DCGM4Net, self).__init__()

        # define initial layer as fully connected
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.n_layers_FFNN = n_layers_FFNN
        self.n_layers_RNN = n_layers_RNN


        # define scaling layer: for standarization

        self.initial_layer_scale = tf.keras.layers.Lambda(lambda x: 2.0*(x - X_low)/(X_high - X_low) - 1.0)
        self.initial_layer = DenseLayer(layer_width, input_dim, transformation = "tanh")

        # define intermediate LSTM layers
        self.LSTMLayerList = []

        for _ in range(self.n_layers_RNN):
            self.LSTMLayerList.append(LSTMLayer(layer_width, input_dim))


        # define intermediate Dense layers

        self.DenseLayerList = []
        for _ in range(self.n_layers_FFNN):
            self.DenseLayerList.append(DenseLayer(
                layer_width, layer_width, transformation=activation_FFNN))
            

        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(
            output_dim, layer_width, transformation=final_trans)

    # main function to be called

    def __call__(self, x):
        '''            
        Args:
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (x)                
        '''

        # define input vector 
        X = x

        # call initial layer scaling
        X = self.initial_layer_scale(X)
        S = self.initial_layer.call(X)
        
        # call intermediate LSTM layers
        for i in range(self.n_layers_RNN):
            S = self.LSTMLayerList[i].call(S, X) 
            
        Inter_S = S
            
        for i in range(self.n_layers_FFNN):
            S = self.DenseLayerList[i].call(S) + S
            
        # call final layers
        result = self.final_layer.call(S+Inter_S)

        return result


