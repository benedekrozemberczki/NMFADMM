import scipy as sp
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy.linalg import inv

class ADMM_NMF:
    """
    Alternating Direction Method of Multipliers for Non-Negative Matrix Factorization
    """
    def __init__(self, V, args):
        """
        Class initialization.
        :param args: Arguments object.
        :param V: Data matrix.
        """
        self.args = args
        self.V = V
        self._init_weights()

    def _init_weights(self):

        self.W = np.random.uniform(-0.1, 0.1, (self.V.shape[0],self.args.dimensions)) 
        self.H = np.random.uniform(-0.1, 0.1, (self.args.dimensions,self.V.shape[1]))
        X_i, Y_i = sp.nonzero(self.V)
        values = np.sum(self.W[X_i]*self.H[:, Y_i].T+np.random.uniform(0,1,(self.args.dimensions,)), axis=-1)
        self.X = sp.sparse.coo_matrix((values, (X_i,Y_i)),shape = self.V.shape)

        self.W_plus = np.random.uniform(0, 0.1, (self.V.shape[0],self.args.dimensions)) 
        self.H_plus = np.random.uniform(0, 0.1, (self.args.dimensions,self.V.shape[1]))

        self.alpha_X = sp.sparse.coo_matrix(([0]*len(values), (X_i,Y_i)), shape = self.V.shape)
        self.alpha_W = np.zeros(self.W.shape)
        self.alpha_H = np.zeros(self.H.shape)
        
    
    def update_W(self):
        left = np.linalg.pinv( self.H.dot(self.H.T)+np.eye(self.args.dimensions))
        right_1 = self.X.dot(self.H.T).T+self.W_plus.T
        right_2 = (1.0/self.args.rho)*(self.alpha_X.dot(self.H.T).T-self.alpha_W.T)
        self.W = left.dot(right_1+right_2).T

    def update_H(self):
        left = np.linalg.pinv(self.W.T.dot(self.W)+np.eye(self.args.dimensions))
        right_1 = self.X.T.dot(self.W).T+self.H_plus
        right_2 = (1.0/self.args.rho)*(self.alpha_X.T.dot(self.W).T-self.alpha_H)
        self.H = left.dot(right_1+right_2)

    def update_X(self):
        iX, iY = sp.nonzero(self.V)
        values = np.sum(self.W[iX]*self.H[:, iY].T, axis=-1)
        scores = sp.sparse.coo_matrix((values-1, (iX,iY)),shape = self.V.shape)
        left = self.args.rho*scores-self.alpha_X
        right = (left.power(2)+4.0*self.args.rho*self.V).power(0.5)
        self.X = (left+right)/(2*self.args.rho)

    def update_W_plus(self):
        self.W_plus = np.maximum(self.W+(1/self.args.rho)*self.alpha_W,0)

    def update_H_plus(self):
        self.H_plus =  np.maximum(self.H+(1/self.args.rho)*self.alpha_H,0)

    def update_alpha_X(self):
        iX, iY = sp.nonzero(self.V)
        values = np.sum(self.W[iX]*self.H[:, iY].T, axis=-1)
        scores = sp.sparse.coo_matrix((values, (iX,iY)),shape = self.V.shape)
        self.alpha_X = self.alpha_X+self.args.rho*(self.X-scores)

    def update_alpha_W(self):
        self.alpha_W = self.alpha_W+self.args.rho*(self.W-self.W_plus)
 
    def update_alpha_H(self):
        self.alpha_H = self.alpha_H+self.args.rho*(self.H-self.H_plus)

    def optimize(self):
        
        for i in tqdm(range(self.args.epochs)):
            self.update_W()
            self.update_H()
            self.update_X()
            self.update_W_plus()
            self.update_H_plus()
            self.update_alpha_X()
            self.update_alpha_W()
            self.update_alpha_H()

    def save_user_factors(self):
        columns = ["id"] + ["x_" + str(x) for x in range(self.args.dimensions)]
        ids = np.array(range(self.V.shape[0])).reshape(-1,1)
        user_factor = np.concatenate([ids,self.W_plus],axis=1)
        user_factor = pd.DataFrame(user_factor, columns = columns)
        user_factor.to_csv(self.args.user_path, index = None)
              
    def save_item_factors(self):
        columns = ["id"] + ["x_" + str(x) for x in range(self.args.dimensions)]
        ids = np.array(range(self.V.shape[1])).reshape(-1,1)
        item_factor = np.concatenate([ids,self.H_plus.T],axis=1)
        item_factor = pd.DataFrame(item_factor, columns = columns)
        item_factor.to_csv(self.args.item_path, index = None)

        

        
