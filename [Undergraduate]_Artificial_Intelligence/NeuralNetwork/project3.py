import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.utils import check_random_state


def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)


# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, train_data, color):
    # Set min and max values and give it some padding
    x_min, x_max = train_data[:, 0].min() - .5, train_data[:, 0].max() + .5
    y_min, y_max = train_data[:, 1].min() - .5, train_data[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=color, cmap=plt.cm.RdYlGn)


class NeuralNetwork(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_hdim3, nn_output_dim, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
            W4: Fourth layer weights
            b4: Fourth layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons in the hidden layer H1.
            nn_hdim2: (int) The number of neurons in the hidden layer H2.
            nn_hdim3: (int) The number of neurons in the hidden layer H3.
            nn_output_dim: (int) The number of classes C.
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random":
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_hdim3)
            self.model['b3'] = np.zeros((1, nn_hdim3))
            self.model['W4'] = np.random.randn(nn_hdim3, nn_output_dim)
            self.model['b4'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_hdim3))
            self.model['b3'] = np.zeros((1, nn_hdim3))
            self.model['W4'] = np.ones((nn_hdim3, nn_output_dim))
            self.model['b4'] = np.zeros((1, nn_output_dim))

    def forward_propagation(self, X):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        
        Args:
            X: Input data of shape (N, D)
            
        Returns:
            y_hat: (numpy array) Array of shape (N,) giving the classification scores for X
            cache: (dict) Values needed to compute gradients
            
        """
        W1, b1, W2, b2, W3, b3, W4, b4 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3'], self.model['W4'], self.model['b4']
        
        ### CODE HERE ###
        # Layer 1
        h1 = np.dot(X, W1) + b1
        z1 = relu(h1)               # ReLU activation

        # Layer 2
        h2 = np.dot(z1, W2) + b2
        z2 = leakyrelu(h2)          # LeakyReLU activation

        # Layer 3
        h3 = np.dot(z2, W3) + b3
        z3 = tanh(h3 + h1)          # Tanh activation

        # Output layer
        h4 = np.dot(z3, W4) + b4
        y_hat = sigmoid(h4)         # Sigmoid activation
        
        y_hat = y_hat.reshape(-1)   # Flatten y_hat, y_hat.shape is (N, 1) but we need (N,)
        # raise NotImplementedError("Erase this line and write down your code.")
        ############################
        
        assert y_hat.shape==(X.shape[0],), f"y_hat.shape is {y_hat.shape}. Reshape y_hat to {(X.shape[0],)}"
        cache = {'h1': h1, 'z1': z1, 'h2': h2, 'z2': z2, 'h3': h3, 'z3': z3, 'h4': h4, 'y_hat': y_hat}
    
        return y_hat, cache

    def back_propagation(self, cache, X, y, L2_norm=0.0):
        """
        Compute the gradients
        
        Args:
            cache: (dict) Values needed to compute gradients
            X: (numpy array) Input data of shape (N, D)
            y: (numpy array) Training labels (N, ) -> (N, 1)
            L2_norm: (int) L2 normalization coefficient
            
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
            
        """
        W1, b1, W2, b2, W3, b3, W4, b4 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3'], self.model['W4'], self.model['b4']
        h1, z1, h2, z2, h3, z3, h4, y_hat = cache['h1'], cache['z1'], cache['h2'], cache['z2'], cache['h3'], cache['z3'], cache['h4'], cache['y_hat']
        
        # For matrix computation
        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)
        
        ### CODE HERE ###
        # L = -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)  # Binary Cross Entropy Loss
        # dL = 1                                            # dL/dL = 1
        
        dy_hat = -(y / y_hat) + (1 - y) / (1 - y_hat)   # dL/dy_hat = -(y / y_hat - (1 - y) / (1 - y_hat)), which is the derivative of Binary Cross Entropy Loss

        dh4 = dy_hat * (1 - y_hat) * y_hat              # dL/dh4, [upstream_grad] * [local_grad] = dy_hat * [(1-sigmoid(x))*sigmoid(x)]
        db4 = np.sum(dh4, axis=0, keepdims=True)        # Add gate: dL/db4 = np.sum(dh4, axis=0), which is same as dh4 but b4 is scalar so we need to sum it
        dW4 = np.dot(z3.T, dh4) + 2 * L2_norm * W4      # Mul gate: dL/dW4 = np.dot(z3.T, dh4) + 2 * L2_norm * W4
        dz3 = np.dot(dh4, W4.T)                         # Mul gate: dL/dz3 = np.dot(dh4, W4.T)
        
        dh3 = dz3 * (1 - z3**2)                         # Derivative of tanh(x) is 1 - tanh(x)^2, dL/dh3 = [upstream_grad] * [local_grad] = dL/dz3 * (1 - z3^2)
        db3 = np.sum(dh3, axis=0, keepdims=True)        # Add gate: dL/db3 = np.sum(dh3, axis=0), which is same as dh3 but b3 is scalar so we need to sum it
        dW3 = np.dot(z2.T, dh3) + 2 * L2_norm * W3      # Mul gate: dL/dW3 = np.dot(z2.T, dh3) + 2 * L2_norm * W3
        dz2 = np.dot(dh3, W3.T)                         # Mul gate: dL/dz2 = np.dot(dh3, W3^T)
        
        dh2 = dz2 * np.where(h2 > 0, 1, 0.01)           # Derivative of leakyrelu(x) is 1 if x > 0 else 0.01, dL/dh2 = [upstream_grad] * [local_grad] = dL/dz2 * np.where(h2 > 0, 1, 0.01)
        db2 = np.sum(dh2, axis=0, keepdims=True)        # Add gate: dL/db2 = np.sum(dh2, axis=0), which is same as dh2 but b2 is scalar so we need to sum it
        dW2 = np.dot(z1.T, dh2) + 2 * L2_norm * W2      # Mul gate: dL/dW2 = np.dot(z1.T, dh2) + 2 * L2_norm * W2
        dz1 = np.dot(dh2, W2.T)                         # Mul gate: dL/dz1 = np.dot(dh2, W2^T)
        
        
        # dh3_h3 = np.ones_like(h3)
        # dh3_z2 = np.dot(dh3_h3, W3.T)
        # dh3_h2 = dh3_z2 * np.where(h2 > 0, 1, 0.01)
        # dh3_dz1 = np.dot(dh3_h2, W2.T)
        # dh3_dh1 = dh3_dz1 * np.where(h1 > 0, 1, 0)
        # dh1 = dh3 * (1 + dh3_dh1)
        dh1 = dh3 * (1 + dz1 * np.where(h1 > 0, 1, 0))
        # dh1 = dz1 * np.where(h1 > 0, 1, 0) + dh3        # Derivative of relu(x) is 1 if x > 0 else 0, be careful with residual connection from h3
                                                        # dL/dh1 = [upstream_grad] * [local_grad] + residual_grad = dL/dz1 * np.where(h1 > 0, 1, 0) + dL/dh3
        db1 = np.sum(dh1, axis=0, keepdims=True)        # Add gate: dL/db1 = np.sum(dh1, axis=0), which is same as dh1 but b1 is scalar so we need to sum it
        dW1 = np.dot(X.T, dh1) + 2 * L2_norm * W1       # Mul gate: dL/dW1 = np.dot(X.T, dh1) + 2 * L2_norm * W1
        # raise NotImplementedError("Erase this line and write down your code.")
        ################
        
        ############################
        
        grads = dict()
        grads['dy_hat'] = dy_hat
        grads['dh4'] = dh4
        grads['dW4'] = dW4
        grads['db4'] = db4
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        
        return grads

    
    def compute_loss(self, y_pred, y_true, L2_norm=0.0):
        """
        Descriptions:
            Evaluate the total loss on the dataset
        
        Args:
            y_pred: (numpy array) Predicted target (N,)
            y_true: (numpy array) Array of training labels (N,)
        
        Returns:
            loss: (float) Loss (data loss and regularization loss) for training samples.
        """
        W1, b1, W2, b2, W3, b3, W4, b4 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3'], self.model['W4'], self.model['b4']
        
        ### CODE HERE ###
        data_loss = - np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))        # Binary Cross Entropy Loss

        # Regularization Loss: L2 norm of all weights
        reg_loss = L2_norm * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))     # L2 regularization loss

        # Total Loss
        total_loss = data_loss + reg_loss
        # raise NotImplementedError("Erase this line and write down your code.")
        ############################

        return total_loss
        

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N,)
            y_pred: (numpy array) Predicted target (N,)
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        for it in range(epoch):
            ### CODE HERE ###
            y_hat, cache = self.forward_propagation(X_train)
            loss = self.compute_loss(y_hat, y_train, L2_norm)
            grads = self.back_propagation(cache, X_train, y_train, L2_norm)
            
            self.model['W1'] -= learning_rate * grads['dW1']
            self.model['b1'] -= learning_rate * grads['db1']
            self.model['W2'] -= learning_rate * grads['dW2']
            self.model['b2'] -= learning_rate * grads['db2']
            self.model['W3'] -= learning_rate * grads['dW3']
            self.model['b3'] -= learning_rate * grads['db3']
            self.model['W4'] -= learning_rate * grads['dW4']
            self.model['b4'] -= learning_rate * grads['db4']

            # for param in self.model:
            #     self.model[param] = self.model[param] - learning_rate * grads[f'd{param}']
            # raise NotImplementedError("Erase this line and write down your code.")
            ################# 
            if (it+1) % 1000 == 0:
                loss_history.append(loss)

                y_train_pred = self.predict(X_train)
                train_acc = np.average(y_train==y_train_pred)
                train_acc_history.append(train_acc)
                
                if X_val is not None:
                    y_val_pred = self.predict(X_val)
                    val_acc = np.average(y_val==y_val_pred)
                    val_acc_history.append(val_acc)

            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss}")
 
        if X_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }
        else:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
            }

    def predict(self, X):
        ### CODE HERE ###
        y_hat, _ = self.forward_propagation(X)
        
        y_hat = y_hat.reshape(-1)              # Flatten y_hat, y_hat.shape is (N, 1) but we need (N,)
        
        prediction = (y_hat > 0.5).astype(int) # Convert y_hat to binary prediction, 1 if y_hat > 0.5 else 0
        
        return prediction
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  



def tanh(x):
    ### CODE HERE ###
    x = np.tanh(x)
    # x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # raise NotImplementedError("Erase this line and write down your code.")
    #################  
    return x


def relu(x):
    ### CODE HERE ###
    x = np.maximum(0, x)
    # raise NotImplementedError("Erase this line and write down your code.")
    ############################
    return x


def leakyrelu(x):
    ### CODE HERE ###
    x = np.maximum(0.01*x, x)
    # raise NotImplementedError("Erase this line and write down your code.")
    ############################
    return x 


def sigmoid(x):
    ### CODE HERE ###
    x = 1/(1+np.exp(-x))
    # raise NotImplementedError("Erase this line and write down your code.")#
    ############################
    return x


######################################################################################



class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear layer.
        
        Args:
            x: (numpy array) Array containing input data, of shape (N, D)
            w: (numpy array) Array of weights, of shape (D, M)
            b: (numpy array) Array of biases, of shape (M,)

        Returns: 
            out: (numpy array) output, of shape (N, M)
            cache: (tupe[numpy array]) Values needed to compute gradients
        """
        ### CODE HERE ###
        out = np.dot(x, w) + b
        cache = (x, w)
        # raise NotImplementedError("Erase this line and write down your code.")
        #################
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for an linear layer.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: (numpy array) Gradient with respect to x, of shape (N, D)
            dw: (numpy array) Gradient with respect to w, of shape (D, M)
            db: (numpy array) Gradient with respect to b, of shape (M,)
        """

        ### CODE HERE ###
        x, w = cache
        dx = np.dot(dout, w.T)
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0) # keepdims=True
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Args:
            x: (numpy array) Input

        Returns:
            out: (numpy array) Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = relu(x)
        cache = (x, out)
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        x, out = cache
        dx = dout * (x > 0)
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return dx


class LeakyReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of LeakyReLUs.

        Args:
            x: (numpy array) Input

        Returns:
            out: (numpy array) Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = leakyrelu(x)
        cache = (x, out)
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of LeakyReLU.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        x, out = cache
        dx = dout * np.where(x > 0, 1, 0.01) 
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return dx


class Tanh(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of Tanh.

        Args:
            x: Input

        Returns:
            out: Output, array of the same shape as x
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = tanh(x)
        cache = (x, out)
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Tanh.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        x, out = cache
        dx = dout * (1 - out**2)
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return dx


class Sigmoid(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of Sigmoid.

        Args:
            x: Input

        Returns:
            out: Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = sigmoid(x)
        cache = (x, out)
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Sigmoid.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        x, out = cache
        dx = dout * out * (1 - out)
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return dx


class SigmoidWithBCEloss(object): 

    @staticmethod
    def forward(x, y=None):
        """
        if y is None, computes the forward pass for a layer of sigmoid with binary cross-entropy loss.
        Else, computes the loss for binary classification.
        Args:
            x: Input data
            y: Training labels or None 
       
        Returns:
            if y is None:
                y_hat: (numpy array) Array of shape (N,) giving the classification scores for X
            else:
                loss: (float) data loss
                cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        y_hat = sigmoid(x)
        y_hat = y_hat.reshape(-1)
        
        if y is None:
            return y_hat
        else:
            loss = - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
            cache = (y_hat, y)
        # raise NotImplementedError("Erase this line and write down your code.")
        ################# 
        assert y_hat.shape==(x.shape[0],), f"y_hat.shape is {y_hat.shape}. Reshape y_hat to {(x.shape[0],)}"
        return loss, cache

    @staticmethod
    def backward(cache, dout=None):
        """
        Computes the loss and gradient for softmax classification.
        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        y_hat, y = cache
        # For matrix computation
        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)
        
        ### CODE HERE ###
        dy_hat = -(y / y_hat) + ((1 - y) / (1 - y_hat))   # dL/dy_hat = -((y / y_hat) - ((1 - y) / (1 - y_hat))), which is the derivative of Binary Cross Entropy Loss
        dx = dy_hat * (1 - y_hat) * y_hat                 # dL/dx = [upstream_grad] * [local_grad] = dy_hat * [(1-sigmoid(x))*sigmoid(x)]
        # raise NotImplementedError("Erase this line and write down your code.")    
        ################# 
        return dx


class Dropout(object):
    
    @staticmethod
    def forward(x, dropout_rate, train=False):
        """
        Computes the forward pass for a layer of Dropout.
        
        Args:
            x: Input data
            dropout_rate: Probability of dropping a neuron active during dropout
            train: (bool) if True, perform dropout
        
        Returns:
            out: Output data
            cache: Values needed for backward pass
        """
        ### CODE HERE ###
        if train:
            mask = (np.random.rand(*x.shape) < 1.0 - dropout_rate) / (1.0 - dropout_rate)   # Create dropout mask: 1 means keep neuron, 0 means drop neuron, scale adjustment
            out = x * mask                                                                  # Apply mask to input
        else:
            mask = None                                                                     # During inference, scale input to account for dropout
            out = x                                                                         # No dropout during testing
        
        cache = mask                                                                        # Store mask for backward pass
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return out, cache
    
    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Dropout.
        
        Args:
            cache: Values needed for backward pass
            dout: Upstream derivatives
        
        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        mask = cache                # Get mask from cache
        if mask is not None:
            dx = dout * mask        # Apply mask to upstream gradient
        else:
            dx = dout               # No dropout during testing
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        return dx


class NeuralNetwork_module(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_hdim3, nn_output_dim, dropout_rate, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons  in the hidden layer H1.
            nn_hdim2: (int) The number of neurons H2 in the hidden layer H1.
            nn_output_dim: (int) The number of classes C.
            dropout_rate: (float) Probability of dropping a neuron active during dropout
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random":
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_hdim3)
            self.model['b3'] = np.zeros((1, nn_hdim3))
            self.model['W4'] = np.random.randn(nn_hdim3, nn_output_dim)
            self.model['b4'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_hdim3))
            self.model['b3'] = np.zeros((1, nn_hdim3))
            self.model['W4'] = np.ones((nn_hdim3, nn_output_dim))
            self.model['b4'] = np.zeros((1, nn_output_dim))
            
        self.dropout_rate = dropout_rate

    def forward(self, X, y=None, train=False):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        
        Args:
            X: Input data of shape (N, D)
            y: (numpy array) Training labels (N,) or None
            
        Returns:
            if y is None:
                y_hat: (numpy array) Array of shape (N,) giving the classification scores for X
            else:
                loss: (float) data loss
                cache: Values needed to compute gradients
            
        """

        W1, b1, W2, b2, W3, b3, W4, b4 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3'], self.model['W4'], self.model['b4']
        cache = {}
        
        ### CODE HERE ###
        h1, cache['Linear1'] = Linear.forward(X, W1, b1)
        z1, cache['ReLU1'] = ReLU.forward(h1)
        z1, cache['Dropout1'] = Dropout.forward(z1, self.dropout_rate, train)
        
        h2, cache['Linear2'] = Linear.forward(z1, W2, b2)
        z2, cache['LeakyReLU2'] = LeakyReLU.forward(h2)
        z2, cache['Dropout2'] = Dropout.forward(z2, self.dropout_rate, train)
        
        h3, cache['Linear3'] = Linear.forward(z2, W3, b3)
        z3, cache['Tanh3'] = Tanh.forward(h3 + h1)
        z3, cache['Dropout3'] = Dropout.forward(z3, self.dropout_rate, train)
        
        out, cache['Linear4'] = Linear.forward(z3, W4, b4)
        
        # raise NotImplementedError("Erase this line and write down your code.")
        #################  
        if y is None:
            y_hat = SigmoidWithBCEloss.forward(out)
            return y_hat
        else: 
            loss, cache['SigmoidWithBCEloss'] = SigmoidWithBCEloss.forward(out, y)
            return cache, loss
    
    def backward(self, cache, L2_norm=0.0):
        """
        Compute the gradients
        
        Args:
            cache: (dict) Values needed to compute gradients
            L2_norm: (int) L2 normalization coefficient
            
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
            
        """
        dh4 = SigmoidWithBCEloss.backward(cache['SigmoidWithBCEloss'])
        ### CODE HERE ###
        dz3, dW4, db4 = Linear.backward(cache['Linear4'], dh4)
        dz3 = Dropout.backward(cache['Dropout3'], dz3)
        dW4 += 2 * L2_norm * self.model['W4']
        
        dh3 = Tanh.backward(cache['Tanh3'], dz3)
        dz2, dW3, db3 = Linear.backward(cache['Linear3'], dh3)
        dz2 = Dropout.backward(cache['Dropout2'], dz2)
        dW3 += 2 * L2_norm * self.model['W3']
        
        dh2 = LeakyReLU.backward(cache['LeakyReLU2'], dz2)
        dz1, dW2, db2 = Linear.backward(cache['Linear2'], dh2)
        dz1 = Dropout.backward(cache['Dropout1'], dz1)
        dW2 += 2 * L2_norm * self.model['W2']
        
        dh1 = ReLU.backward(cache['ReLU1'], dz1) + dh3              # Be careful with residual connection from h3
        dx, dW1, db1 = Linear.backward(cache['Linear1'], dh1)
        dW1 += 2 * L2_norm * self.model['W1']
        
        # raise NotImplementedError("Erase this line and write down your code.")
        ###########################################
        grads = dict()
        grads['dW4'] = dW4
        grads['db4'] = db4
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1

        return grads

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True, momentum=0.0):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N, )
            y_pred: (numpy array) Predicted target (N,)
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization
            momentum: (float) Scalar giving momentum strength for optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        if momentum == 0.0:
            optimizer = GradientDescent(learning_rate=learning_rate)
        else:
            optimizer = Momentum(learning_rate=learning_rate, rho=momentum)

        for it in range(epoch):
            ### CODE HERE ###
            cache, loss = self.forward(X_train, y_train, train=True)
            grads = self.backward(cache, L2_norm)
            self.model = optimizer.step(self.model, grads)
            
            # raise NotImplementedError("Erase this line and write down your code.")
            ################# 
            if (it+1) % 1000 == 0:
                loss_history.append(loss)
                
                y_train_pred = self.predict(X_train)
                train_acc = np.average(y_train==y_train_pred)
                train_acc_history.append(train_acc)
                
                if X_val is not None:
                    y_val_pred = self.predict(X_val)
                    val_acc = np.average(y_val==y_val_pred)
                    val_acc_history.append(val_acc)
                    
            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss}")

         
        if X_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }
        else:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
            }

    def predict(self, X):
        ### CODE HERE ###
        y_hat = self.forward(X)
        prediction = (y_hat > 0.5).astype(int)
        return prediction
        # raise NotImplementedError("Erase this line and write down your code.")#
        #################  


class GradientDescent(object):
    def __init__(self, learning_rate=0.01):
        """
        Args:
            learning_rate: (float) Scalar giving learning rate for optimization
        Returns:
        """
        self.learning_rate = learning_rate
        
    def step(self, params:dict, grads:dict):
        """
        Args:
            params: (dict) Dictionary mapping parameter names to arrays of parameter values
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
        Returns:
            params: (dict) Dictionary mapping parameter names to updated parameter values
        """
        ### CODE HERE ###
        for key in params.keys():
            params[key] -= self.learning_rate * grads[f'd{key}']
        # raise NotImplementedError("Erase this line and write down your code.")
        #################
        return params


class Momentum(object):
    def __init__(self, learning_rate=0.01, rho=0.9):
        """
        Args:
            learning_rate: (float) Scalar giving learning rate for optimization
            rho: (float) Scalar giving momentum strength
        Returns:
        """
        self.learning_rate = learning_rate
        self.rho = rho
        self.velocity = None
        
    def step(self, params:dict, grads:dict):
        """
        Args:
            params: (dict) Dictionary mapping parameter names to arrays of parameter values
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
        Returns:
            params: (dict) Dictionary mapping parameter names to updated parameter values
        """
        ### CODE HERE ###
        if self.velocity is None:
            self.velocity = {key: np.zeros_like(value) for key, value in params.items()}
    
        for key in params.keys():
            self.velocity[key] = self.rho * self.velocity[key] + grads[f'd{key}']  # Momentum update
            params[key] -= self.learning_rate * self.velocity[key]                 # Parameter update
        # raise NotImplementedError("Erase this line and write down your code.")        
        #################
        return params