import numpy as np

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

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
    
    