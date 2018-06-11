# -*- coding: utf-8 -*-
"""
Spyder Editor
Girl Birth Example of Chapt 2 at page 37
This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
# -----------------------
# import logit and inverse-logit, also known as logistic or expit
# logit(p)= log(p/(1-p))
# logistic(y)= 1/(1+exp(-y))
# ----------------------
from scipy.special import logit, expit
from scipy.stats import beta
#import itertools as it

a,b=438,544
sample_size=1000
# --------------------------------------------
# Draw samples from the posterior distribution
# --------------------------------------------
samples=np.random.beta(a,b,sample_size)

# ----------------------------------
# Plot histogram of samples
# ----------------------------------
def plot_hist(samples,title):
    n=len(samples)
    plt.hist(samples, bins=50, histtype='bar' )
    plt.title(str(n)+' samples from '+title)
    plt.ylabel('y-value')

plot_hist(samples, 'posterior beta distribution')


def summary_stat(samples):
    posterior_mean=samples.mean()
    posterior_std=samples.std()
    posterior_median=np.median(samples)
    posterior_interval=np.percentile(samples, [2.5,97.5])
    return(posterior_mean, posterior_std, posterior_median, posterior_interval)


post_mean, post_std, post_median, post_interval=summary_stat(samples)


# -----------------------------------------
# Normal approximation of interval
# Difference comparing to direct approximation, e.g. without logit 
# is noticable if sample size is small or 
# the distribution of p includes value 0 or 1 
# -----------------------------------------
def norm_appx_int(sample):
    natural_samples=logit(samples)
    plot_hist(natural_samples,'logit-transformed samples')
    logit_mean, logit_std, logit_median, logit_interval=summary_stat(natural_samples)
    normal_logit_interval=np.array([logit_mean-1.95*logit_std, logit_mean+1.95*logit_std])
    normal_appx_int=expit(normal_logit_interval)
    return(normal_appx_int)
    
norm_logit_int=norm_appx_int(samples)


# -------------------
# plot beta pdf 
# -------------------

def alpha_beta(mean, appx_sample_size):
    a=mean*appx_sample_size
    b=appx_sample_size - a
    return(a,b)
    
sum=np.array([2,5,10,20,100,200])
mean=0.485

a_array, b_array = alpha_beta(mean, sum)  
ind=0  
for a,b in zip(a_array,b_array):
    print (a,b)
    ind+=1
    x=np.linspace(0,1,100)
    #x = np.linspace(beta.ppf(0.01, a, b),
    #              beta.ppf(0.99, a, b), 100)
    
    axes=plt.subplot(3,2,ind)
    axes.set_xlim([0,1])
    axes.set_ylim([0,16])
    plt.ylabel('f(x)')
    plt.xlabel('x')
    plt.plot(x, beta.pdf(x, a, b),
             'r-', lw=3, alpha=0.4, label='beta pdf')
    plt.text(0.1,10,('a='+str(a),'b='+str(b)))

plt.tight_layout()
plt.suptitle('pdf of Beta(a,b)')





















