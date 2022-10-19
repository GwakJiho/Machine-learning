import numpy as np
from collections import OrderedDict
from Activation import Affine, Sigmoid, Relu,  SoftmaxWithLoss
   
class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, hidden_size1, hidden_size2, output_size, weight_init_std=0.01):
        # 가중치 초기화, 계층의 구조(크기) 입력, weight_init_std: 가중치의 표준편차 설정
        self.input_size = input_size
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size1)
        self.params['b2'] = np.zeros(hidden_size1)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b3'] = np.zeros(hidden_size2)
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size2, output_size)
        self.params['b4'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):  # x.shape: (A, input_size), t.shape: (A, 1), A: 배치사이즈
        y = self.predict(x)       # 미니배치 predict 결과 행렬, y.shape: (A, output_size)
        y = np.argmax(y, axis=1)  # np.argmax(y, axis=1): 행 기준 최대값 인덱스 (A, )
        
        accuracy = np.sum(y == t) / float(x.shape[0])  # predict=정답인 횟수를 A로 나눔
        return accuracy
    
    def numerical_gradient(self, f, x):
        h = 1e-4 # 0.0001
        grad = np.zeros_like(x)
    
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x) # f(x+h)

            x[idx] = tmp_val - h 
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)

            x[idx] = tmp_val # 값 복원
            it.iternext()   
        
        return grad

    def gradient(self, x, t):
        self.loss(x, t)
        
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db
        
        return grads # dictionary 형

        