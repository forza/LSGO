from __future__ import division
import numpy as np

class Benchmark_2013(object):
    def __init__(self, index):
        if index in [1, 4, 7, 8, 11, 12, 13, 14, 15]:
            self.ub = 100
            self.lb = -100
        elif index in [2, 5, 9]:
            self.ub = 5
            self.lb = -5
        elif index in [3, 6, 10]:
            self.ub = 32
            self.lb = -32
        else:
            raise ValueError('Error: There are only 15 functions in this test suite!')
      
        self.x_opt = np.loadtxt('./datafiles/F%d-xopt.txt' % index)
        self.alpha = 10
        self.beta = 0.2
        self.m = 5
        self.s = None
        self.p = None
        self.r25 = None
        self.r50 = None
        self.r100 = None

        if index in range(4, 15):
            if index == 12 or index == 15:
                pass
            else:
                self.s = np.loadtxt('./datafiles/F%d-s.txt' % index)
                self.r25 = np.loadtxt('./datafiles/F%d-R25.txt' % index, delimiter=',')
                self.r50 = np.loadtxt('./datafiles/F%d-R50.txt' % index, delimiter=',')
                self.r100 = np.loadtxt('./datafiles/F%d-R100.txt' % index, delimiter=',')
                self.w = np.loadtxt('./datafiles/F%d-w.txt' % index)
                self.p = np.loadtxt('./datafiles/F%d-p.txt' % index)

    # The six base functions
    
    # The sphere function
    @staticmethod
    def _sphere(x):
        return np.inner(x, x)

    # The elliptic function
    @staticmethod
    def _elliptic(x):
        condition = 1e+6 ** np.linspace(0, 1, x.size)
        return np.inner(condition, x**2)
    
    # The rastrigin's function
    @staticmethod
    def _rastrigin(x):
        return np.inner(x, x) + 10*x.size - 10*np.cos(2*np.pi*x).sum()
    
    # The ackley's function
    @staticmethod
    def _ackley(x):
        return -20*np.exp(-0.2*np.sqrt(np.inner(x, x)/x.size)) - np.exp(np.cos(2*np.pi*x).sum()/x.size) + 20 + np.e
    
    # The schwefel's function
    @staticmethod
    def _schwefel(x):
        fit = 0
        for i in range(x.size):
            fit += (x[0:i+1].sum()) ** 2
        return fit
    
    # The rosenbrock's function
    @staticmethod
    def _rosenbrock(x):
        a = x[0:x.size-1]**2 - x[1:x.size]
        b = x - 1
        return 100*np.inner(a, a) + np.inner(b, b)
    
    ''' Helper functions for the function transformation'''
    # A transformation function to create smooth local irregularities
    @staticmethod
    def _t_osz(x):
        indx = (x>0)
        x_osz = x.copy()
        x_osz[indx] = np.log(x_osz[indx])
        x_osz[indx] = np.exp(x_osz[indx] + 0.049*(np.sin(10*x_osz[indx]) + np.sin(7.9*x_osz[indx])))
        indx = (x<0)
        x_osz[indx] = np.log(-1*x_osz[indx])
        x_osz[indx] = -1*np.exp(x_osz[indx] + 0.049*(np.sin(5.5*x_osz[indx]) + np.sin(3.1*x_osz[indx])))
        return x_osz
    
    # A transformation function to break the symmetry of the symmetric functions
    @staticmethod
    def _t_asy(x, beta):
        indx = (x>0)
        power = 1 + beta*(np.linspace(0, 1, x.size)[indx])*np.sqrt(x[indx])
        x_asy = x.copy()
        x_asy[indx] = x[indx]**power
        return x_asy
        
    # This matrix is used to create ill-conditioning
    @staticmethod
    def _t_diag(x, alpha):
        temp = alpha**(0.5*np.linspace(0, 1, x.size))
        return np.diag(temp, 0)
    
    # x_opt is the shift vector to change the location of the global optimum
    @staticmethod
    def _shift(x, x_opt):
        return x - x_opt
    
    ''' The benchmark functions'''
    ''' Fully-separable functions'''
    # Shifted elliptic function
    def f1(self, x):
        return _elliptic(_t_osz(_shift(x, self.x_opt)))
    
    # Shifted rastrigin's function
    def f2(self, x):
        mat = _t_diag(x, self.alpha)
        return _rastrigin(np.dot(mat, _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)))
    
    # Shifted ackley's function
    def f3(self, x):
        mat = _t_diag(x, self.alpha)
        return _ackley(np.dot(mat, _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)))
    
    ''' partially additice separable functions 1'''
    # 7-nonseparable, 1-separable shifted and rotated elliptic function
    def f4(self, x):
        z = _t_osz(_shift(x, self.x_opt))
        count = 0
        fit = 0
        for i in range(self.s.size-1):
            if self.s[i] == 25:
                fit += self.w[i]*_elliptic(np.dot(self.rota_mat_25, z[self.p[count:count+25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i]*_elliptic(np.dot(self.rota_mat_50, z[self.p[count:count+50]]))
                count += 50
            else:
                fit += self.w[i]*_elliptic(np.dot(self.rota_mat_100, z[self.p[count:count+100]]))
                count += 100
        fit += _elliptic(z[self.p[count:])
        return fit
    
    # 7-nonseparable, 1-separable shifted and rotated rastrigin's function
    def f5(self, x):
        z = np.dot(_t_diag(x, self.alpha), _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta))
        count = 0
        fit = 0
        for i in range(self.s.size-1):
            if self.s[i] == 25:
                fit += self.w[i]*_rastrigin(np.dot(self.rota_mat_25, z[self.p[count:count+25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i]*_rastrigin(np.dot(self.rota_mat_50, z[self.p[count:count+50]]))
                count += 50
            else:
                fit += self.w[i]*_rastrigin(np.dot(self.rota_mat_100, z[self.p[count:count+100]]))
                count += 100
        fit += _rastrigin(z[self.p[count:])
        return fit
        
    # 7-nonseparable, 1-separable shifted and rotated ackley's function
    def f6(self, x):
        z = np.dot(_t_diag(x, self.alpha), _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta))
        count = 0
        fit = 0
        for i in range(self.s.size-1):
            if self.s[i] == 25:
                fit += self.w[i]*_ackley(np.dot(self.rota_mat_25, z[self.p[count:count+25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i]*_ackley(np.dot(self.rota_mat_50, z[self.p[count:count+50]]))
                count += 50
            else:
                fit += self.w[i]*_ackley(np.dot(self.rota_mat_100, z[self.p[count:count+100]]))
                count += 100
        fit += _ackley(z[self.p[count:])
        return fit
        
    # 7-nonseparable, 1-separable shifted and rotated schwefel's function
    def f7(self, x):
        z = _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)
        count = 0
        fit = 0
        for i in range(self.s.size-1):
            if self.s[i] == 25:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_25, z[self.p[count:count+25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_50, z[self.p[count:count+50]]))
                count += 50
            else:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_100, z[self.p[count:count+100]]))
                count += 100
        fit += _schwefel(z[self.p[count:])
        return fit
    
    ''' partially additice separable functions 2'''
    # 20-nonseparable shifted and rotated elliptic function   
    def f8(self, x):
        z = _t_osz(_shift(x, self.x_opt))
        count = 0
        fit = 0
        for i in range(self.s.size):
            if self.s[i] == 25:
                fit += self.w[i]*_elliptic(np.dot(self.rota_mat_25, z[self.p[count:count+25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i]*_elliptic(np.dot(self.rota_mat_50, z[self.p[count:count+50]]))
                count += 50
            else:
                fit += self.w[i]*_elliptic(np.dot(self.rota_mat_100, z[self.p[count:count+100]]))
                count += 100
        return fit
        
    # 20-nonseparable shifted and rotated rastrigin's function  
    def f9(self, x):
        z = np.dot(_t_diag(x, self.alpha), _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta))
        count = 0
        fit = 0
        for i in range(self.s.size):
            if self.s[i] == 25:
                fit += self.w[i]*_rastrigin(np.dot(self.rota_mat_25, z[self.p[count:count+25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i]*_rastrigin(np.dot(self.rota_mat_50, z[self.p[count:count+50]]))
                count += 50
            else:
                fit += self.w[i]*_rastrigin(np.dot(self.rota_mat_100, z[self.p[count:count+100]]))
                count += 100
        return fit  
    
    # 20-nonseparable shifted and rotated ackley's function
    def f10(self, x):
        z = np.dot(_t_diag(x, self.alpha), _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta))
        count = 0
        fit = 0
        for i in range(self.s.size):
            if self.s[i] == 25:
                fit += self.w[i]*_ackley(np.dot(self.rota_mat_25, z[self.p[count:count+25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i]*_ackley(np.dot(self.rota_mat_50, z[self.p[count:count+50]]))
                count += 50
            else:
                fit += self.w[i]*_ackley(np.dot(self.rota_mat_100, z[self.p[count:count+100]]))
                count += 100
        return fit
    
    # 20-nonseparable shifted schwefel's function
    def f11(self, x):
        z = _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)
        count = 0
        fit = 0
        for i in range(self.s.size):
            if self.s[i] == 25:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_25, z[self.p[count:count+25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_50, z[self.p[count:count+50]]))
                count += 50
            else:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_100, z[self.p[count:count+100]]))
                count += 100
        return fit
        
    ''' overlapping functions'''
    # shifted rosenbrock's function
    def f12(self, x):
        return _rosenbrock(_shifet(x, self.x_opt))
    
    # shifted schwefel's function with conforming overlapping subcomponents
    def f13(self, x):
        z = _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)
        count = 0
        fit = 0
        for i in range(self.s.size):
            if self.s[i] == 25:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_25, z[self.p[count:count+25]]))
                count += 25-self.m
            elif self.s[i] == 50:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_50, z[self.p[count:count+50]]))
                count += 50-self.m
            else:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_100, z[self.p[count:count+100]]))
                count += 100-self.m
        return fit
    
    # shifted schwefel's function with confllicting overlapping subcomponents
    def f14(self, x):
        c = np.cumsum(self.s)
        count = 0
        fit = 0
        for i in range(self.s.size):
            z = x[self.p[count:count+s[i]]] - self.x_opt(c[i-1]:c[i])
            z = _t_asy(_t_osz(z), self.beta)
            if self.s[i] == 25:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_25, z))
                count += 25-self.m
            elif self.s[i] == 50:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_50, z))
                count += 50-self.m
            else:
                fit += self.w[i]*_schwefel(np.dot(self.rota_mat_100, z))
                count += 100-self.m
        return fit

    ''' fully non-separable functions'''
    # shifted schwefel's function
    def f15(self, x):
        z = _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)
        return _schwefel(z)
    