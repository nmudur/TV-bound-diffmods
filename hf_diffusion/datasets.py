import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
import torch.distributions as dbn

class GaussianMixtureDataset():
    def __init__(self, weights, means, trils, seed=23):
        print(len(weights), len(means), len(trils))
        assert len(weights)==len(means)
        self.Ncomp =len(weights)
        self.dim = len(means[0])
        self.weights = [w.astype(np.float32) for w in weights/np.sum(weights)]
        self.means = [m.astype(np.float32) for m in means]
        self.trils = [c.astype(np.float32) for c in trils]
        self.rng = np.random.default_rng(seed)
        #w, m, tril were all numpy arrays
        self.dbns = [dbn.MultivariateNormal(torch.from_numpy(self.means[c]), scale_tril=torch.from_numpy(self.trils[c])) for c in range(self.Ncomp)]

    def __getitem__(self, index):
       noise = self.rng.standard_normal(size=self.dim, dtype=np.float32)
       
       if self.Ncomp==1:
           return self.means[0] + self.trils[0]@noise.astype(np.float32)
       else:
           index_choice = self.rng.choice(self.Ncomp, p=self.weights)
           return self.means[index_choice] + self.trils[index_choice]@noise
    
    def __len__(self):
        return 500
    
    def get_prob(self, x):
        #q(x)
        lp = 0
        for i in range(self.Ncomp):
            lp += self.weights[i]*np.exp(self.dbns[i].log_prob(x).numpy())
        return lp
    

    
    def get_score(self, x):
        #x must be tensor
        score=0
        for i in range(self.Ncomp):
            qi = torch.exp(self.dbns[i].log_prob(x))
            diff = (x - self.dbns[i].mean).reshape((-1, 1))
            scterm = self.weights[i]*qi*(-torch.linalg.inv(self.dbns[i].covariance_matrix)@diff)
            score += scterm
        return score.numpy()/self.get_prob(x)
    
    def get_partial_derivative_dbns(self, x, ic, d, noqi=False):
        #del q_i/del xd
        qi = np.exp(self.dbns[ic].log_prob(x).numpy())
        pri = torch.linalg.inv(self.dbns[ic].covariance_matrix)
        diff = (x - self.dbns[ic].mean)
        if d==0:
            dimterm = pri[0,0]*diff[0] + 0.5*(pri[0, 1]+pri[1, 0])*diff[1]
        if d==1:
            dimterm = pri[1, 1]*diff[1] + 0.5*(pri[0, 1]+pri[1, 0])*diff[0]
        if noqi:
            return -dimterm #-(P00s0 + (P01 + P10)0.5*s1)
        return -qi*(dimterm) #-qi(P00s0 + (P01 + P10)0.5*s1) for d=0
    
    def get_partial_derivative_gmm(self, x, d, noqi=False):
        #del q_(x)/del xd
        pd = 0
        for i in range(self.Ncomp):
            pd+=self.weights[i]*self.get_partial_derivative_dbns(x, i, d, noqi=noqi)#-sum_i w_i del_qi/del x_d
        return pd
    
    def get_partial_derivative_score(self, x, d, tol = 1e-5):
        #score = self.get_score(x)
        
        qx = self.get_prob(x)
        if qx<1e-5:
            ans=0
            for i in range(self.Ncomp):
                idbn = self.dbns[i]
                iprec = idbn.precision_matrix 
                idiff = x - idbn.mean 
                diffterm = (iprec@idiff)[d] #P@(x-mu)
                term1 = iprec[d, d]
                term2 = self.get_partial_derivative_dbns(x, i, d, noqi=True)*diffterm
                term3 = -self.get_partial_derivative_gmm(x, d)*diffterm
                print(term1, term2, term3)
                ans += -self.weights[i]*(term1+term2+term3)

        else:
            ans =0

            for i in range(self.Ncomp):
                idbn = self.dbns[i]
                iprec = idbn.precision_matrix
                idiff = x - idbn.mean
                diffterm = (iprec@idiff)[d] #P@(x-mu)
                term1 = np.exp(idbn.log_prob(x))*(iprec[d, d])
                term2 = self.get_partial_derivative_dbns(x, i, d)*diffterm
                term3 = -np.exp(idbn.log_prob(x))*self.get_partial_derivative_gmm(x, d)*diffterm/qx
                print(term1, term2, term3)
                ans += -self.weights[i]*(term1+term2+term3)/qx
        return ans
    
    
            

if __name__=='__main__':
    weights = np.array([0.7, 0.3], dtype=np.float32)
    means = [np.array([-1, 2], dtype=np.float32), np.array([2, -1], dtype=np.float32)]
    trils = [np.diag(np.array([0.8, 0.8], dtype=np.float32)), np.diag(np.array([0.6, 0.6], dtype=np.float32))]
    data = GaussianMixtureDataset(weights=weights, means=means, trils=trils)
    print(data[34], data[34].dtype)
