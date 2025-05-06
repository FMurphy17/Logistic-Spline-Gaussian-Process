#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pymc.gp.cov import Covariance
from pymc.gp.mean import Mean
import pymc as pm
import pytensor.tensor as tt
from sklearn.linear_model import LinearRegression
import arviz as az 

class Heteroskedastic(pm.gp.cov.Covariance):
    ## adapted from pm.gp.cov.WhiteNoise
    def __init__(self, sigmas):
        super().__init__(1, None)
        self.sigmas = tt.as_tensor_variable(sigmas)
        
    def diag(self, X):
        return self.sigmas
    
    def full(self, X, Xs=None):
        if Xs is None:
            return tt.diag(self.diag(X))
        else:
            return tt.alloc(0.0, X.shape[0], Xs.shape[0])
        
class Wiener(Covariance):
    r"""
    The Wiener Kernel of order q
    """

    def __init__(self, input_dim, q=1, active_dims=None):
        super().__init__(input_dim, active_dims)
        self.q=q

    def full(self, X, Xs=None):
        if Xs is None:
            Xs = X 
        X1 = tt.tile(X,reps=Xs.shape[0])
        X2 = tt.tile(Xs,reps=X.shape[0]).T
        if self.q==1:
            return pm.math.minimum(X1,X2)
        else:
            return ((X1**2)/2*(X2-X1/3)*(X1<=X2))+((X2**2)/2*(X1-X2/3)*(X2<X1))

    def diag(self, X):
        if self.q==1:
            return X
        else:
            return (X**2)/2*(X-X/3)

# The warp function is equal to the underlying parametric model
def warp_func(X, theta1, theta2, b):
    '''
    X should be a column vector
    '''
    Xtime  = X[:,[0]] #time
    Xother = X[:, 1:] # other covariates
    theta1x = pm.math.exp(pm.math.matmul(Xother,theta1))
    bx = pm.math.matmul(Xother,theta2)-pm.math.exp(pm.math.matmul(Xother,b))*Xtime
    return  theta1x*pm.math.invlogit(-bx) #theta1x/(1+pm.math.exp(bx))

# The mean function of the Gaussian Process is equal to the warp function
class sigmmean(Mean):
    
    def __init__(self, theta1=1, theta2=1, b=1):
        Mean.__init__(self)
        self.theta1 = theta1
        self.theta2 = theta2
        self.b = b

    def __call__(self, X): 
        return warp_func(X,self.theta1,self.theta2,self.b)[:,0]


# Logistic Spline Gaussian Process Model
class modelGP:
    
    def __init__(self, dataframe,add_intercept=True,heteroskedastic=False):
        self.expkey = np.unique(dataframe['index_experiment'])
        self.nexp=len(self.expkey)
        self.Xcolnames = [col for col in dataframe.columns if col.startswith('X_')]
        self.Ycolnames = [col for col in dataframe.columns if col.startswith('Y_')]
        self.timecol = dataframe['time']
        self.dataframe = dataframe
        self.modelPyMC = []
        self.add_intercept=add_intercept
        self.heteroskedastic = heteroskedastic

    def fit(self,num_restarts=1, eps = 1e-5, theta1mean=np.log(100)
            ,theta2mean=np.log(76.5625)
            ,bmean=np.log(0.196)): 
        
        if self.add_intercept==True:
            dimx = len(self.Xcolnames)+1
            mutheta1=np.zeros(dimx)
            mutheta1[-1]=theta1mean
            mutheta2=np.zeros(dimx)
            mutheta2[-1]=theta2mean
            mub=np.zeros(dimx)
            mub[-1]=bmean
            
            sigmaparam=np.ones(dimx)*5.0 
            sigmaparam[-1]=8.0 #1

        else:
            dimx = len(self.Xcolnames)
            sigmaparam=np.ones(dimx)*5.0 
            mutheta1=np.zeros(dimx)
            mutheta2=np.zeros(dimx)
            mub=np.zeros(dimx)

        map_params = []
        for map_iter in range(num_restarts):
            
            with pm.Model() as model:
                theta1 = pm.Normal("theta1",  mu=mutheta1, sigma=sigmaparam, shape=(dimx,1), initval= 0.1*np.random.randn(dimx,1))
                theta2 = pm.Normal("theta2",  mu=mutheta2, sigma=sigmaparam, shape=(dimx,1), initval= 0.1*np.random.randn(dimx,1)) 
                b = pm.Normal("b",  mu=mub, sigma=sigmaparam, shape=(dimx,1), initval= 0.1*np.random.randn(dimx,1)) 
                
                sigmakunc = pm.HalfCauchy.dist(5)
                sigmak = pm.Truncated("sigmak",sigmakunc,lower=1e-4) 
                #sigmak = pm.Truncated("sigmak", sigmakunc, lower=1e-8, initval=1e-4) 
            
                noise_a=[]
                noise_b=[]
                cov_func=[]
                gp=[]
                f=[]

                for i in range(self.nexp): 
                    # Covariance function    
                    cov_func.append(pm.gp.cov.WarpedInput(dimx+1, warp_func=warp_func, 
                                                     args=(theta1, theta2, b), cov_func=Wiener(1,q=2))) 
                    cov_white = pm.gp.cov.WhiteNoise(eps) # this is added to avoid numeric problem
                    
                    # Specify the GP.  
                    gp.append(pm.gp.Marginal(mean_func=sigmmean(theta1,theta2,b), 
                                             cov_func=sigmak*cov_func[i]+cov_white))
                    
                    df = self.dataframe[self.dataframe['index_experiment'] == self.expkey[i]] # select data for experiment ind 
                    Xother = df[self.Xcolnames].astype(float).values
                    if self.add_intercept==True:
                        Xother = np.hstack([Xother,np.ones((Xother.shape[0],1))]) #add intercept
                    #add time
                    XX = np.hstack([df['time'].astype('float').values[:,None],Xother])
                    XXa = XX
                    yya = df[self.Ycolnames[0]].astype('float').values[:,None]
                    for j in range(1,len(self.Ycolnames)):
                        yy = df[self.Ycolnames[j]].astype('float').values[:,None]
                        yya = np.hstack([yya,yy])
                        XXa = np.vstack([XXa,XX])
                    logsigma = np.log(np.var(yya,axis=1)+1e-3)
                    reg = LinearRegression().fit(df['time'].astype('float').values[:,None],logsigma)
                    if self.heteroskedastic==True:
                        a_mean = reg.intercept_
                        b_mean= reg.coef_[0]
                    else:
                        a_mean=np.mean(logsigma)
                        b_mean=0.0
                    
                    noise_a.append(
                        pm.Normal('a_noise_'+str(i), mu=a_mean, sigma = 1.25, initval= a_mean+0.1*np.random.randn(1)[0]) 
                        ) 
                    noise_b.append(
                        pm.Normal('b_noise_'+str(i), mu=b_mean, sigma = 1.25, initval = 0.01*np.random.randn(1)[0])
                        )
                    if self.heteroskedastic==True:
                        sigmvec = pm.Deterministic('sigma^2_noise_'+str(i), 0.01+pm.math.exp(noise_a[i]+noise_b[i]*XXa[:,0]))#time dependent nosie variance
                    else:
                        sigmvec = pm.Deterministic('sigma^2_noise_'+str(i), 0.01+pm.math.exp(noise_a[i]*tt.ones(XXa.shape[0])))#constant noise variance
                    noise = Heteroskedastic(sigmvec)
                    f.append(gp[i].marginal_likelihood("f_"+str(i), X=XXa, y=yya.T.flatten(), sigma=noise)) # noise
                if num_restarts > 1:
                    try:
                        map_params.append(pm.find_MAP(maxeval=10000,return_raw=True)) 
                    except Exception as e:
                        print(e)
                else:
                    map_params.append(pm.find_MAP(maxeval=10000,return_raw=True)) 
            
        max_val = -map_params[0][1]["fun"]
        self.mp = map_params[0][0]
        for p in map_params:
            if -p[1]["fun"] > max_val:
                max_val = -p[1]["fun"]
                self.mp = p[0]
        print(self.mp)
        self.modelPyMC = model
        self.gp = gp
          
    # Predictions from the GP
    def predict(self, dfpred, nsamples=2000, gp_marginal = None):
        expkey = np.unique(dfpred['index_experiment'])
        pred_samples_dic={}
        for exp in expkey:
            df = dfpred[dfpred['index_experiment'] == exp]#select data for experiment ind 
            Xother = df[self.Xcolnames].astype(float)
            if self.add_intercept==True:
                Xother = np.hstack([Xother,np.ones((Xother.shape[0],1))]) #add intercept
            #add time
            XX = np.hstack([df['time'].astype('float').values[:,None],Xother])
            iexp = np.where(exp == self.expkey)[0][0] 
            try:
                with self.modelPyMC:
                    f_pred = self.gp[iexp].conditional("f_pred_"+str(iexp), XX, jitter=1e-3)
                    gp_mean_cov = self.gp[iexp].predict(XX, self.mp) 
            except Exception as error:
                print("An exception occurred:", type(error).__name__, "–", error) # An exception occurred
            with self.modelPyMC:
                pred_samples = pm.sample_posterior_predictive([self.mp]*nsamples, var_names=["f_pred_"+str(iexp)]) 
            if self.heteroskedastic==True:
                sigma_noise = pm.math.exp(self.mp['a_noise_'+str(iexp)]+self.mp['b_noise_'+str(iexp)]*XX[:,0]).eval()
            else:
                sigma_noise = np.exp(self.mp['a_noise_'+str(iexp)]*np.ones(XX.shape[0]))
            rand_noise=np.random.randn(nsamples,len(sigma_noise))*np.sqrt(sigma_noise)
            fsamples = az.extract(pred_samples, group="posterior_predictive", var_names=["f_pred_"+str(iexp)])
            pred_samples={}
            pred_samples["f_pred_"+str(iexp)]=fsamples.T
            pred_samples["y_pred_"+str(iexp)]=pred_samples["f_pred_"+str(iexp)]+rand_noise
            pred_samples_dic[exp]=pred_samples
        return pred_samples_dic, gp_mean_cov
        