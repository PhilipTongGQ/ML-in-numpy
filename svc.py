import numpy as np
import time
import copy

#support vector classifier with various descent methods
class svc:                                                                          #an implementation using numpy                     
    def __init__(self,method="sgd",C=100,init_weights="random",tol=0.001,learning_rate=0.0001,max_iterations=np.inf,batch_size=None,get_cost=False):
        self.method=method
        self.init_weights=init_weights
        self.tol=tol
        self.learning_rate=learning_rate
        self.max_iterations=max_iterations
        self.C=C
        self.batch_size=batch_size                                                  #only active for mini_batch method
        self.w=None
        self.get_cost=get_cost
        self.cost_values=[]
    
    def check_input(self,X,y):
        if isinstance(X,np.ndarray) and isinstance(y,np.ndarray):
            return
        else:
            raise TypeError("X and y should be numpy arrays")

    def cost(self,X,y):
        c=0.5*np.sum(self.w**2)+self.C*np.sum(np.maximum(0,1-y*X@self.w))      #dimension of y*X@self.w is n*1
        if self.get_cost:
            self.cost_values.append(c)
        return c

    def sgd_update(self,i,j,X,y):
        w=self.w[j]
        correct=y[i]*self.w.T@X[i]>=1                                               #only one observation for sgd
        if correct:
            update=0
        else:
            update=-y[i]*X[i][j]
        return w+self.C*update if j!=0 else self.C*update

    def sgd(self,X,y):
        n,d=X.shape
        train_X,train_y=copy.deepcopy(X),copy.copy(y)                               #so that we don't shuffle original data
        i,k=0,0
        delta=0                                                                     #make sure the loop starts
        cost=self.cost(train_X,train_y)                                             #initialize cost
        while True:
            ind=np.random.permutation(X.shape[0])
            train_X,train_y=train_X[ind],train_y[ind]                               #shuffle both X and y
            for j in range(d):
                self.w[j]-=self.learning_rate*self.sgd_update(i,j,train_X,train_y)
            i=i%n+1 if i!=n-1 else 0                                                #if i==n-1, it will cause index error, it should be set to 0
            k+=1        
            if k>=self.max_iterations:
                print("maximum iterations achieved")
                break
            new_cost=self.cost(train_X,train_y)
            delta_pct=np.abs(cost-new_cost)*100/cost
            delta=0.5*delta+0.5*delta_pct
            cost=new_cost
            if delta<self.tol:
                break
        return k
    
    def batch_update(self,j,X,y):
        update=np.sum(np.where(y*X@self.w>=1,0,-y.flatten()*X[:,j]))                                            
        return self.w[j]+self.C*update if j!=0 else self.C*update

    def batch_descent(self,X,y):
        n,d=X.shape
        k,delta=0,0
        cost=self.cost(X,y) 
        while True:
            for j in range(d):
                self.w[j]-=self.learning_rate*self.batch_update(j,X,y)
            k+=1
            if k>=self.max_iterations:
                print("maximum iterations achieved")
                break
            new_cost=self.cost(X,y)
            delta_pct=np.abs(cost-new_cost)*100/cost
            delta=0.5*delta+0.5*delta_pct
            cost=new_cost
            if delta<self.tol:
                break
        return k
    
    def mini_batch_update(self,l,j,X,y):
        w=self.w[j]
        batch_range=range((l*self.batch_size+1),min(X.shape[0],(l+1)*self.batch_size))
        update=np.sum(np.where(y[batch_range]*X[batch_range]@self.w>=1,0,-y[batch_range].flatten()*X[batch_range,j]))
        return w+self.C*update if j!=0 else self.C*update

    def mini_batch_descent(self,X,y):
        if self.batch_size is None:
            raise ValueError("for mini-batch descent, batch_size must be specified")
        n,d=X.shape
        train_X,train_y=copy.deepcopy(X),copy.copy(y)
        k,delta=0,0
        l=0
        cost=self.cost(train_X,train_y)
        while True:
            ind=np.random.permutation(X.shape[0])
            train_X,train_y=train_X[ind],train_y[ind] 
            for j in range(d):
                self.w[j]-=self.learning_rate*self.mini_batch_update(l,j,train_X,train_y)
            k+=1
            l=int((l+1)%((n+self.batch_size-1)/self.batch_size))
            if k>=self.max_iterations:
                print("maximum iterations achieved")
                break
            new_cost=self.cost(X,y)
            delta_pct=np.abs(cost-new_cost)*100/cost
            delta=0.5*delta+0.5*delta_pct
            cost=new_cost
            if delta<self.tol:
                break
        return k

    def fit(self,X,y):    
    #X is columns of independent variables. X and y must be numpy arrays
        self.check_input(X,y)                                           #needs to be np.arrays
        X=np.hstack((np.ones(X.shape[0])[:,np.newaxis],X))              #built in intercept
        if self.init_weights=="random":
            self.w=np.random.randn(X.shape[1])                          
        elif self.init_weights=="zeros":
            self.w=np.zeros(X.shape[1])
        else:
            raise ValueError("Unkown initialization method")
        start=time.time()
        if self.method=="sgd":
            iterations=self.sgd(X,y)
            end=time.time()
            print("stochastic gradient descent finished with {} iterations, took {} seconds ".format(iterations,np.round(end-start,2)))
        elif self.method=="batch":
            iterations=self.batch_descent(X,y)
            end=time.time()
            print("batch gradient descent finished with {} iterations, took {} seconds ".format(iterations,np.round(end-start,2)))
        elif self.method=="mini-batch":
            iterations=self.mini_batch_descent(X,y)
            end=time.time()
            print("mini-batch gradient descent finished with {} iterations, took {} seconds ".format(iterations,np.round(end-start,2)))
        else:
            raise ValueError("Unkown descent method")
        return self

    def predict(self,X):
        if self.w is None:
            raise ValueError("need to fit the model first")
        return np.sign(np.hstack((np.ones(X.shape[0])[:,np.newaxis],X))@self.w)
