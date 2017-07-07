import numpy as np

class FTRL_ADP:
    def __init__(self, decay, L1, L2, LP, adaptive = False, n_inputs = 1):
        self.ADAPTIVE  = adaptive
        
        self.L1 = L1
        self.L2 = L2
        self.LP = LP
        self.v = np.zeros(n_inputs)
        self.h = np.zeros(n_inputs)
        self.z = np.zeros(n_inputs)
        
        self.r = 1
        self.d = 1 / (self.L2 + self.LP*self.r)
        self.decay = decay
        
        self.times = 0
        self.fails = 0
        self.times_warn = 0
        self.fails_warn = 0        
        self.p_min = 1.
        self.s_min = 10.

    def fit(self, idx, x, y):     
        # Make prediction
        w = self.weight_update(idx)        
        p = self.__sigmoid(np.dot(w, x))
        
        # Update decay rate
        if self.ADAPTIVE:
            self.times += 1
            self.fails += int(np.abs(y-p)>0.5)
            
            #self.decay = (np.cbrt(self.times) - 1)/np.cbrt(self.times)
            #self.decay = (np.sqrt(self.times) - 1)/np.sqrt(self.times)
            #self.decay = float(self.times - 1)/self.times
            #self.decay = float(self.times)/(self.times+1)
            self.decay = 1. - np.log(self.times)/(2 * self.times)

            if self.times > 30:
                p_i = float(self.fails)/self.times
                s_i = np.sqrt(p_i*(1-p_i)/self.times)
                ps = p_i + s_i
                
                self.decay = self.decay * 0.99 + 0.01*p_i

                if ps < self.p_min + self.s_min: # Remember the (p,s) with minimum sum
                    self.p_min = p_i
                    self.s_min = s_i
                    
                if ps < self.p_min + 2*self.s_min:
                    self.times_warn = 0
                    self.fails_warn = 0
                else:
                    self.times_warn += 1
                    self.fails_warn += int(np.abs(y-p)>0.5)
                    
                    if ps > self.p_min + 3*self.s_min:
                        self.times = self.times_warn
                        self.fails = self.fails_warn
                        self.p_min = 1.
                        self.s_min = 10.   
                        
        # Update parameter
        g = (p - y)*x
        
        self.v[idx] = self.v[idx] + g
        self.h[idx] = self.decay*self.h[idx] + w
        self.z[idx] = self.v[idx] - self.LP*self.h[idx]
        self.r = 1 + self.decay*self.r
                        
        return p, self.decay
    
    def weight_update(self, idx):
        w = np.zeros(idx.size)
        mask = np.abs(self.z[idx]) > self.L1        
        
        z_i = self.z[idx][mask]
        
        tmp_1_ = z_i - self.L1*np.sign(z_i)
        tmp_2_ = self.L2 + self.LP*self.r     
        
        w[mask] = -np.divide(tmp_1_, tmp_2_)         
        
        return w
        
    def predict(self, idx, x):
        w = self.weight_update(idx)
        
        return self.__sigmoid(np.dot(w, x)) 
    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) 