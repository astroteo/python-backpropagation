import numpy as np
import csv
import matplotlib.pyplot  as plt
from numpy import genfromtxt
from sklearn.preprocessing import OneHotEncoder
import os
from random import randint
from PIL import Image, ImageDraw
import pickle as pkl
import pandas as pd
import math
from sklearn.model_selection import train_test_split

class Ann:
    def __init__(self, layers_size,layers_activations, loss ="binary_cross_entropy", optimizer = "SGD", momentum = 0.9, epsilon = 1e-8):
        self.layers_size = layers_size

        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
        self.accs = []

        self.layers_activations = layers_activations
        assert(len(self.layers_activations) == self.L-1)
        self.activations  = {}
        self.activations_derivatives =  {}

        self.activations[str(0)] = self.no_act
        self.activations_derivatives[str(0)] =self.no_act_derivative

        # set loss functions
        for i,activation in enumerate(layers_activations):
            if activation == "sigmoid":
                self.activations[str(i+1)] = self.sigmoid
                self.activations_derivatives[str(i+1)] = self.sigmoid_derivative
            elif activation == "tanh":
                self.activations[str(i+1)] = self.tanh
                self.activations_derivatives[str(i+1)] = self.tanh_derivative

            elif activation == "relu":
                self.activations[str(i+1)] = self.relu
                self.activations_derivatives[str(i+1)] = self.relu_derivative


            else:
                self.activations[str(i+1)] = self.no_act
                self.activations_derivatives[str(i+1)] = self.no_act

        # set loss and optimizer
        self.loss = loss
        self.optimizer = optimizer


        if self.optimizer != "SGD":
            self.optim_dic = {}
            self.momentum = momentum


    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self,Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def tanh(self,Z):
        return np.tanh(Z)

    def no_act(self,Z):
        return Z

    def relu(self,Z):
        Z_rel = np.zeros(Z.shape)
        for i in range( Z.shape[0]):
            for j in range( Z.shape[1]):
                z = Z[i,j]
                Z_rel[i,j] = z if  z > 0 else 0

        return Z_rel


    def initialize_parameters(self):
        np.random.seed(1)

        ##TODO: add options depending on activations
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))

    def initialize_optimizer(self):
        if self.optimizer == "MOMENTUM" or self.optimizer == "NESTEROV":
            for l in range(1,len(self.layers_size)):
                self.optim_dic["dv_W" + str(l)] = np.zeros(self.parameters["W" + str(l)].shape)
                self.optim_dic["dv_b" + str(l)] = np.zeros(self.parameters["b" + str(l)].shape)
        else:
            pass

    def forward(self, X):
        store = {}
        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.activations[str(l)](Z)

            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z

        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]

        if self.loss == 'binary_cross_entropy':
            A = self.sigmoid(Z)
        elif self.loss == "cross_entropy":
            A = self.softmax(Z)
        else:#regressor problem
            A = Z;

        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z

        return A, store

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def tanh_derivative(self,Z):
        return  1 - np.power(np.tanh(Z),2)

    def no_act_derivative(self,Z):
        return Z

    def relu_derivative(self,Z):
        Z_rel = np.zeros(Z.shape)

        for i in range( Z.shape[0]):
            for j in range( Z.shape[1]):
                z = Z[i,j]
                Z_rel[i,j] = 1 if  z > 0 else 0

        return Z_rel

    def backward(self, X, Y, store):

        derivatives = {}

        store["A0"] = X.T

        A = store["A" + str(self.L)]
        if self.loss == 'binary_cross_entropy':
            dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A)
            dZ = dA * self.sigmoid_derivative(store["Z" + str(self.L)])
        elif self.loss == 'L2':
            dZ = 2 * (A - Y.T)
        elif self.loss == "cross_entropy":
            dZ = A - Y.T
            #TODO: Replace as dedicated fucntion softmax_derivative


        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):
            #dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            #dZ = dAPrev * self.tanh_derivative(store["Z"+str(l)])
            dZ = dAPrev * self.activations_derivatives[str(l)](store["Z"+str(l)])

            dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives

    def fit(self, X, Y, learning_rate=0.01, n_iterations=2500, filename = None):

        # Add input layer getting dimensions from
        self.layers_size.insert(0, X.shape[1])

        # Initialize parameters
        self.initialize_parameters()

        # Initialize parameters
        self.initialize_optimizer()

        print("start-training-------------------------------------------------")
        print("optimizer: "+self.optimizer)
        np.random.seed(1)

        self.n = X.shape[0]

        self.layers_size.insert(0, X.shape[1])

        for loop in range(n_iterations):
            A, forward_pass = self.forward(X)

            if self.loss == "binary_cross_entropy":
                cost = np.squeeze(-(Y.dot(np.log(A.T)) + (1 - Y).dot(np.log(1 - A.T))) / self.n)

            elif self.loss == "cross_entropy":
                cost = -np.mean(Y.T * np.log(A + 1e-8))
                                                # 1e-8 to avoid issue with log(0)

            else:
                cost = np.mean(np.power((Y-A),2))

            if self.optimizer != "MOMENTUM" and self.optimizer != "SGD":
                for l in range(1, self.L + 1):
                    dv_W = self.optim_dic["dv_W" + str(l)]
                    dv_b = self.optim_dic["dv_b" + str(l)]
                    self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - dv_W
                    self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - dv_b
            else:
                pass

            derivatives = self.backward(X, Y, forward_pass)

            if self.optimizer == "SGD":
                for l in range(1, self.L + 1):
                    self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                        "dW" + str(l)]
                    self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                        "db" + str(l)]

            elif self.optimizer == "MOMENTUM" or self.optimizer == "NESTEROV":
                for l in range(1, self.L + 1):
                    dv_W = self.optim_dic["dv_W" + str(l)]
                    dv_b = self.optim_dic["dv_b" + str(l)]

                    d_W = derivatives["dW" + str(l)]
                    d_b = derivatives["db" + str(l)]

                    dv_W = self.momentum * dv_W + ( 1- self.momentum ) * d_W
                    dv_b = self.momentum * dv_b + (1- self.momentum ) * d_b

                    self.optim_dic["dv_W" + str(l)] = dv_W
                    self.optim_dic["dv_b" + str(l)] = dv_b

                    self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * dv_W
                    self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * dv_b


            if loop % 10 == 0:
                self.costs.append(cost)
                _ , acc = self.predict(X,Y)
                self.accs.append(acc)

                print("Epoch: " + str(loop )+" / ".strip()+ str(n_iterations) + " Accuracy: "+ str(acc) + " Cost: " + str(cost))

        if filename:
            with open(filename,'wb') as f:
                pkl.dump(self.parameters, f)
            print("trained parameters stored at: " + filename)

        print("end-training-------------------------------------------------")

    def predict(self, X, Y = None):
        A, _ = self.forward(X)

        n = X.shape[0]
        p = np.zeros(n)

        if self.loss == "binary_cross_entropy":

            for i in range(0, A.shape[1]):
                if A[0, i] > 0.5:
                    p[i] = 1
                else:
                    p[i] = 0
            ##TODO: solve the nested issue here.

        elif self.loss == "cross_entropy":
            p = np.argmax(A, axis=0)
            if Y is not None:
                Y = np.argmax(Y, axis=1) #One hot encoded

        else:
            p = A

        #print(p)
        #print(Y)
        acc =0
        if Y is not None:
            if self.loss == "cross_entropy" or self.loss == "binary_cross_entropy":

                for pred, ground_truth in zip(p,Y):
                    if pred == ground_truth:
                        acc += 1
            else:
                for pred, ground_truth in zip(p[0],Y[0]):
                    #acc += np.sqrt((pred - ground_truth)**2)
                    acc += (pred - ground_truth)**2
            return  pred,acc/n

        else:
            pred = p

            return pred

    def load_train(self,filename):
        with open(filename, 'rb') as b_W:
            self.parameters = pkl.load(b_W)

    def plot_cost(self):
        plt.figure(1)
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()

    def plot_acc(self):
        plt.figure(2)
        plt.plot(np.arange(len(self.accs)), self.accs)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.show()


#From file to np.array, Mnist {8,5}
def get_binary_dataset():
    #bp = "C://Users/BAMA306/Documents/Datascience/".replace()
    bp = os.getcwd() + " / ".strip()
    train_x = genfromtxt(bp + 'Xbinary_train.csv', delimiter=',')
    train_y = genfromtxt(bp + 'Ybinary_train.csv', delimiter=',')
    test_x = genfromtxt(bp + 'Xbinary_test.csv', delimiter=',')
    test_y = genfromtxt(bp + 'Ybinary_test.csv', delimiter=',')

    train_x = train_x.T
    test_x = test_x.T

    return train_x, train_y, test_x, test_y

#From file to np.array, Mnist
def get_full_dataset(one_hot_encoded = True):
    #bp = "C://Users/BAMA306/Documents/Datascience/".replace()
    bp = os.getcwd() + " / ".strip()
    train_x = genfromtxt(bp + 'Xfull_train.csv', delimiter=',')
    train_y = genfromtxt(bp + 'Yfull_train.csv', delimiter=',')
    test_x = genfromtxt(bp + 'Xfull_test.csv', delimiter=',')
    test_y = genfromtxt(bp + 'Yfull_test.csv', delimiter=',')

    train_x = train_x.T
    test_x = test_x.T

    return train_x, train_y, test_x, test_y

#Scale [0,1]
def pre_process_inputs(train_x, test_x):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    return train_x, test_x

#One-Hot encoder
def pre_process_labels(train_y,test_y):
    enc = OneHotEncoder(sparse=False)

    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
    test_y = enc.transform(test_y.reshape(len(test_y), -1))
    return train_y, test_y

# scaler to get scaled signals (non numeric columns are removed)
def scale_ds(df,mode = "min_max" ):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    df = df.select_dtypes(include=numerics)
    df = df.dropna(thresh = len(df.columns))
    df_s = pd.DataFrame([])
    for c in df.columns:
        m_ = df[c].min()
        M_ = df[c].max()
        avg = df[c].mean()
        std = df[c].std()
        if mode =="min_max":
            df_s[c] = df[c].apply(lambda x: (x- m_)/(M_-m_))

        elif mode=="std_avg":
            df_s[c] = df[c].apply(lambda x: (x- avg)/std)
    return df_s

def to_numpy(df):
    cols = 0
    try:
        cols = len(df.columns)
    except:
        pass

    if cols:
        matrx = np.zeros((df.count()[0] +1 ,cols))
        print("samples: "+ str((df.count()[0] +1)) + " cols: "+ str(cols))
        for j,c in enumerate(df.columns):
            for i,r in enumerate(df[c]):
                matrx[i,j] = r

    else:
        matrx = np.zeros((df.count()[0] +1 ,))
        for i,r in enumerate(df[c]):
            matrx[i] = r

    return matrx
if __name__ == '__main__':
    ############################################################################
    # select desired test(s)
    binary_mnist = False
    full_mnist = True
    regressor = False

    ############################################################################
    if binary_mnist:
        train_x, train_y, test_x, test_y = get_binary_dataset()
    elif full_mnist:
        train_x, train_y, test_x, test_y = get_full_dataset()
    elif regressor:
        df_train = pd.read_csv(os.getcwd()+"/HousePricing/train.csv")
        df_test = pd.read_csv(os.getcwd()+"/HousePricing/test.csv")

        #for c in df_train.columns:
        #    if c not in df_test.columns:
        #        print("tobe removed from TRAIN",c)
        #        df_train = df_train.drop(c,axis = 1)

        print(sorted(df_train.columns))

        #df_train = scale_ds(df_train,"std_avg")
        df_train = scale_ds(df_train,"min_max")



        # remove id
        df_train = df_train.drop('Id',axis =1)


        train_y = pd.DataFrame(df_train.loc[:,'SalePrice'])
        train_x = df_train.drop('SalePrice',axis =1 )

        train_x = to_numpy(train_x)
        train_y = to_numpy(train_y)
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


                # False , False for mnist bainary from .csv


    pre_process_label_flag = False
    if full_mnist:
        pre_process_label_flag = True

    if pre_process_label_flag:
        train_y, test_y = pre_process_labels(train_y, test_y)

    pre_process_input_flag = False
    if pre_process_input_flag:
        train_x, test_x = pre_process_inputs(train_x, test_x)

    if regressor:
        pass


    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    print("train_y's shape: " + str(train_y.shape))
    print("test_y's shape: " + str(test_y.shape))

    if binary_mnist:
        n_classes = 2
        layers_dims = [196,50,n_classes - 1]
                        #output neuron(s) are left to cost definition
                        # this to avoid inconsistencies.

        layers_activations = ['relu','sigmoid']

        ann = Ann(layers_dims,layers_activations,optimizer = "NESTEROV")
        train = True

        if train:
            ann.fit(train_x, train_y, learning_rate=0.1, n_iterations=1000, filename ="./params_binary.pickle")
            ann.plot_cost()
            ann.plot_acc()
        else:
            ann.load_train("./params_binary.pickle")

        _,train_acc = ann.predict(train_x, train_y)
        _,test_acc = ann.predict(test_x, test_y)

        print("@ binary problem: ")
        print("final accuracy (train): " + str(train_acc))
        print("final accuracy (test): " + str(test_acc))

        n_test_visualize = 10
        dic_lab = {0: 5, 1: 8}

        for _ in range(n_test_visualize):
            idx = randint(0,test_x.shape[1])
            img_vis = test_x[idx,:].reshape((20,20)) * 255
            img_vis = img_vis.T
            img = Image.fromarray(img_vis)
            draw = ImageDraw.Draw(img)

            p_val = ann.predict(test_x[idx,:].reshape(1,20*20))

            t_val = test_y[idx]

            #draw.text((0, 0),"P: " +str(dic_lab[int(p_val[0])]) + " GT: "+ str(dic_lab[t_val]),255)
            print("predicted: " + str(dic_lab[int(p_val[0])]) + " ground-truth: " +str(dic_lab[t_val]))
            img.show()

    elif full_mnist:

        n_classes = 10
        layers_dims = [196,50,n_classes] #TODO: make it clearer => last layer is the classifier, defined as cost, and first layer is automatically detected by input size
        layers_activations = ['sigmoid','tanh']

        ann = Ann(layers_dims,layers_activations,loss ="cross_entropy",optimizer = "NESTEROV")
        train = True
        if train:
            ann.fit(train_x, train_y, learning_rate=0.1, n_iterations=300,filename ="./params_full.pickle")
            ann.plot_acc()
        else:
            ann.load_train("./params_full.pickle")

        _,train_acc = ann.predict(train_x, train_y)
        _,test_acc = ann.predict(test_x, test_y)
        print( "train-set accuracy: "+str(train_acc))
        print( "test-set accuracy: "+str(test_acc))

        n_test_visualize = 10
        for _ in range(n_test_visualize):
            idx = randint(0,test_x.shape[1])
            img_vis = test_x[idx,:].reshape((20,20)) * 255
            img_vis = img_vis.T
            img = Image.fromarray(img_vis)
            draw = ImageDraw.Draw(img)

            p_val = ann.predict(test_x[idx,:].reshape(1,20*20))
            t_val = np.argmax(test_y[idx])


            print("predicted: " + str(p_val) + " ground-truth: " +str(t_val))
            #draw.text((0, 0),"P: " +str(p_val) + " GT: "+ str(t_val),255)
            img.show()

    elif regressor:#regression problem

        layers_dims = [50,25,10,1] #TODO: make it clearer => last layer is the classifier, defined as cost last layer size is one (as regressor)
        layers_activations = ['relu','tanh','sigmoid']
        Train =True
        print(train_x.shape)
        ann = Ann(layers_dims,layers_activations,loss ="L2",optimizer = "SGD")
        if Train:
            ann.fit(train_x, train_y, learning_rate=0.01, n_iterations=3000,filename ="./HousePricing/params_full.pickle")
            ann.plot_cost()
            ann.plot_acc()

        else:
            ann.load_train("./HousePricing/params_full.pickle")

        print(train_y.shape)
        plt.figure(1)
        plt.plot(np.arange(train_y.shape[0]),train_y,'o')
        plt.xlabel("...")
        plt.ylabel("prices")
        plt.show()




        n_test_visualize = 10
        for _ in range(n_test_visualize):
            idx = randint(0,test_x.shape[1])
            x = test_x[idx,:]
            xt = train_x[idx,:]
            yt = train_y[idx]
            y = test_y[idx]
            print(x.shape)
            yp = ann.predict(x.reshape(1,36))
            ypt = ann.predict(xt.reshape(1,36))
            print("test-----")
            print("pred" + str(yp[0]))
            print("gt: "+ str(y[0]))
            print("-----")
            print("train-----")
            print("pred" + str(ypt[0]))
            print("gt: "+ str(yt[0]))
            print("-----")
            #print("predicted value is : " + str(yp) + " exact value is: "+ str(y) + " ==>  error: " + str(math.sqrt((y-yp)**2)) + " and %error: " + str(math.sqrt((y-yp)**2))/y)
