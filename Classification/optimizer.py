  
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
    