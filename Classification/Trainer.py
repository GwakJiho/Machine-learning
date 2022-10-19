import numpy as np
from optimizer import Adam, SGD
from Activation import Affine, Sigmoid, Relu,  SoftmaxWithLoss
from ThreeLayerNet import ThreeLayerNet
from Normalization import BatchNormalization
from OverModel import ThreeLayerNetExtended, ThreeLayerNetExtendedWD

class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=30, mini_batch_size=10,  #default 10
                 optimizer="SGD", optimizer_param={'lr':0.001}, verbose=True):
        
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size

        # optimzer
        optimizer_class_dict = {'sgd':SGD,'adam':Adam}
        model_class_dict = {"none": ThreeLayerNet, "batchnormalization" : ThreeLayerNetExtended, "weigthdecay" : ThreeLayerNetExtendedWD}
        self.network = model_class_dict[network.lower()](784, 50, 30, 20, 10)
        #print(self.network.input_size)
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        
        #if self.verbose: 
         #   print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
                            
            train_acc = self.network.accuracy(self.x_train, self.t_train)
            test_acc = self.network.accuracy(self.x_test, self.t_test)
            
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: 
                print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")

        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))