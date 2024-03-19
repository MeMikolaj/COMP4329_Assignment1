# Load the packages
import numpy as np
import matplotlib.pyplot as plt

# Our data has 10 different labels 0-9 so we need to have 10 neurons as the output (I checked the label data)
# Training data is numbers between -10, 10. Each row has 128 values

# Loop Through:
# 1. Forwards Pass
# 2. Calculate the Loss
# 3. Optimizer 0 grad
# 4. Loss Backwards
# 5. Optimizer Step


# Create an activation class
class Activation(object):

    # ReLU activation function
    def _ReLU(self, x):
        return np.max(np.array([0, x]))
    
    def _ReLU_deriv(self, a):
        # For simplicity, derivative at a=0 is 0 <- very rare event 
        return 1 if a > 0 else 0
        
    # Leaky ReLU activation function - Adjust alpha (0.1)
    def _LeakyReLU(self, x):
        return x if x >= 0 else 0.1*x
    
    def _LeakyReLU_deriv(self, a):
        # For simplicity, derivative at a=0 is 0 <- very rare event
        return 1 if a > 0 else 0.1 # 0.1 = alpha

    # ------------------------------------------ Implement SoftMAX - the last activation function of a neural network to normalize the output and give probabilities  
    
    # ------------------------------------------ Implement GELU
        
    def __init__(self, activation='relu'):
        if activation == 'relu':
            self.f = self._ReLU
            self.f_deriv = self._ReLU_deriv
        elif activation == 'leakyrelu':
            self.f = self._LeakyReLU
            self.f_deriv = self._LeakyReLU_deriv
        else:
            quit()

# ----------------------------------------------------------------------------------------------------------------------------------------------
            
# Define Hidden Layer - WE NEED MORE THAN ONE
class HiddenLayer(object):

    def __init__(self, n_in, n_out, activation_last_layer='relu', activation='relu', W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input=None
        self.activation=Activation(activation).f

        # activation derivative of last layer
        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).f_deriv

        # we randomly assign small values for the weights as the initiallization
        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )

        # we set the size of bias as the size of output dimension
        self.b = np.zeros(n_out,)

        # we set the size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    # Forward Pass
    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b # input * weight + bias
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input=input
        return self.output

    # Backpropagation
    def backward(self, delta, output_layer=False):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta)) # input * delta
        self.grad_b = delta
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input) # delta = delta*weight * derivative_activation
        return delta

# ----------------------------------------------------------------------------------------------------------------------------------------------
    
# Implement MLP with a fully configurable number of layers and neurons. Adaptt the weights using backpropagation algorithmx
class MLP:
    """
    """

    # for initiallization, the code will create all layers automatically based on the provided parameters.
    def __init__(self, layers, activation=[None,'relu','relu']):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used.
        """
        ### initialize layers
        self.layers=[]
        self.params=[]

        self.activation=activation
        # 2 layers: 1 hidden layer. 3 layers: 2 hidden layers.... 
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1]))

    # forward progress: pass the information through the layers and out the results of final output layer
    def forward(self,input):
        for layer in self.layers:
            output=layer.forward(input)
            input=output
        return output

    # ------------------------------------------------------------------------------- Add SoftMax and Cross Entropy Loss, Change 2 methods below
    # define the objection/loss function, we use mean sqaure error (MSE) as the loss
    # you can try other loss, such as cross entropy.
    # when you try to change the loss, you should also consider the backward formula for the new loss as well!
    def criterion_MSE(self,y,y_hat):
        activation_deriv=Activation(self.activation[-1]).f_deriv
        # MSE
        error = y-y_hat
        loss=error**2
        # calculate the MSE's delta of the output layer
        delta=-2*error*activation_deriv(y_hat)
        # return loss and delta
        return loss,delta

    # backward progress
    def backward(self,delta):
        delta=self.layers[-1].backward(delta,output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta)

    # update the network weights after backward.
    # make sure you run the backward function before the update function!
    def update(self,lr):
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b


    # ------------------------------------------------ Here last layer output we need to do Softmax, then cross-entropy loss
    # define the training function
    # it will return all losses within the whole training process.
    def fit(self,X,y,learning_rate=0.1, epochs=100):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """
        X=np.array(X)
        y=np.array(y)
        to_return = np.zeros(epochs)

        for k in range(epochs):
            loss=np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i=np.random.randint(X.shape[0])

                # forward pass
                y_hat = self.forward(X[i])

                # backward pass
                loss[it],delta=self.criterion_MSE(y[i],y_hat)
                self.backward(delta)
                y # --------------------------- y what? Incomplete code form tutorial
                # update
                self.update(learning_rate)
            to_return[k] = np.mean(loss)
        return to_return

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i,:])
        return output
    




####################  - LEARNING AND TESTING -  ####################
    
### Try different MLP models # 2 Hidden Layers here
nn = MLP([80, 40 ,10], [None,'relu','leakyrelu'])
# input_data = 
# output_data = 

### Try different learning rate and epochs
#MSE = nn.fit(input_data, output_data, learning_rate=0.001, epochs=500)
#print('loss:%f'%MSE[-1])

# Visualise the loss!!!
#plt.figure(figsize=(15,4))
#plt.plot(MSE)
#plt.grid()

### ---------------- Try different MLP models, different hyperparameters

# Testing!!!!!!!!!!!!!!!
# output = nn.predict(input_data)
# visualizing the predict results
# notes: since we use tanh function for the final layer, that means the output will be in range of [0,1]
#plt.figure(figsize=(8,6))
#plt.scatter(output_data, output, s=100)
#plt.xlabel('Targets')
#plt.ylabel('MLP output')
#plt.grid()


# ------------- Add graph with ground truth data and with our predictions on test data


# From Tutorial 4. Adam. 
def gd_adam(df_dx, x0, conf_para=None):

    if conf_para is None:
        conf_para = {}

    conf_para.setdefault('n_iter', 1000) #number of iterations
    conf_para.setdefault('learning_rate', 0.001) #learning rate
    conf_para.setdefault('rho1', 0.9)
    conf_para.setdefault('rho2', 0.999)
    conf_para.setdefault('epsilon', 1e-8)

    x_traj = []
    x_traj.append(x0)
    t = 0
    s = np.zeros_like(x0)
    r = np.zeros_like(x0)

    for iter in range(1, conf_para['n_iter']+1):
        dfdx = np.array(df_dx(x_traj[-1][0], x_traj[-1][1]))
        t += 1
        s = conf_para['rho1']*s + (1-conf_para['rho1'])*dfdx # 1
        r = conf_para['rho2']*r + (1-conf_para['rho2'])*(dfdx**2) # 2
        st = s / (1 - conf_para['rho1']**t) # 3.1
        rt = r / (1 - conf_para['rho2']**t) # 3.2

        x_traj.append(x_traj[-1] - conf_para['learning_rate'] * st / (np.sqrt(rt+conf_para['epsilon'])) * dfdx) # 4 - modified solution from tutorial, epsilon also in sqrt

    return x_traj

# Need to incorporate it in here:
x0 = np.array([1.0,1.5]) # Initial point - We need to figure out how to choose one
conf_para_momentum = {'n_iter':100,'learning_rate':0.005} # parameters
x_traj_momentum = gd_adam(dbeale_dx, x0, conf_para_momentum) # dbeale_dx is a derivative of a surface function which we optimise on, with respect to x1 and x2 returning dfdx1, dfdx2
print("The final solution is (x_1,x_2) = (",x_traj_momentum[-1][0],",",x_traj_momentum[-1][1],")") #The answer that we found