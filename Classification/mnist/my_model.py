import numpy as np
from collections import OrderedDict


class Sigmoid:
    def __init__(self):
        self.out = None

    def sigmoid(self, x):  
        return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))     
    
    def forward(self, x):
        out = self.sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b        
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def cross_entropy_error(self, y, t):   # y, t : 배치 입력
        if y.ndim == 1:   # .dim -> numpy 배열의 차원 구하기, 1차원 배열이면
            t = t.reshape(1, t.size) # 2차원 배열로 바꾸기
            y = y.reshape(1, y.size)
                                                     
        if t.size == y.size:
            t = t.argmax(axis=1)
                                                             
        batch_size = y.shape[0]      # 행 갯수 확인 -> 배치 사이즈
        return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0) 
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T  # 행렬 원래대로 전치
                                                             
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))    
        
    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
    
    
    
class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화, 계층의 구조(크기) 입력, weight_init_std: 가중치의 표준편차 설정
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # np.random.randn(): Gaussian 표준 정규분포(0,1)로 랜덤 값 생성
        # 초기값의 표준편차를 줄임으로써 gradient descent 값이 안정적으로 나오도록 함
        # 가중치의 차원은 [현재 층의 노드(뉴런) 수, 다음 층의 노드(뉴런) 수]가 됨
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
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
        
        return grads # dictionary 형
    
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]

class Adam:
    def __init__(self, lr=0.001, B1 = 0.9, B2 = 0.999):
        self.lr = lr
        self.B1 = B1
        self.B2 = B2
        self.iter = 0
        self.m  = None
        self.v = None
    
    def update(self, params, grads):
        if self.m is None:
            self.m , self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.B2**self.iter) / (1.0 - self.B1**self.iter)
        
        for key in params.keys():
            self.m[key] += (1 - self.B1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.B2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
    
        