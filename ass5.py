#The PyStan Model


import numpy as np
import matplotlib.pyplot as plt
import pystan
import seaborn as sns
import statistics as st



"""
Spyder Editor

This is a temporary script file.
"""


y = [607, 583, 521, 494, 369, 782, 570, 678, 467, 620, 425, 395, 346, 361, 310, 300, 382, 294, 315, 323, 421, 339, 398, 328, 335, 291, 329, 310, 294, 321, 286, 349, 279, 268, 293, 310, 259, 241, 243, 272, 247, 275, 220, 245, 268, 357, 273, 301, 322, 276, 401, 368, 149, 507, 411, 362, 358, 355, 362, 324, 332, 268, 259, 274, 248, 254, 242, 286, 276, 237, 259, 251, 239, 247, 260, 237, 206, 242, 361, 267, 245, 331, 357, 284, 263, 244, 317, 225, 254, 253, 251, 314, 239, 248, 250, 200, 256, 233, 427, 391, 331, 395, 337, 392, 352, 381, 330, 368, 381, 316, 335, 316, 302, 375, 361, 330, 351, 186, 221, 278, 244, 218, 126, 269, 238, 194, 384, 154, 555, 387, 317, 365, 357, 390, 320, 316, 297, 354, 266, 279, 327, 285, 258, 267, 226, 237, 264, 510, 490, 458, 425, 522, 927, 555, 550, 516, 548, 560, 545, 633, 496, 498, 223, 222, 309, 244, 207, 258, 255, 281, 258, 226, 257, 263, 266, 238, 249, 340, 247, 216, 241, 239, 226, 273, 235, 251, 290, 473, 416, 451, 475, 406, 349, 401, 334, 446, 401, 252, 266, 210, 228, 250, 265, 236, 289, 244, 327, 274, 223, 327, 307, 338, 345, 381, 369, 445, 296, 303, 326, 321, 309, 307, 319, 288, 299, 284, 278, 310, 282, 275, 372, 295, 306, 303, 285, 316, 294, 284, 324, 264, 278, 369, 254, 306, 237, 439, 287, 285, 261, 299, 311, 265, 292, 282, 271, 268, 270, 259, 269, 249, 261, 425, 291, 291, 441, 222, 347, 244, 232, 272, 264, 190, 219, 317, 232, 256, 185, 210, 213, 202, 226, 250, 238, 252, 233, 221, 220, 287, 267, 264, 273, 304, 294, 236, 200, 219, 276, 287, 365, 438, 420, 396, 359, 405, 397, 383, 360, 387, 429, 358, 459, 371, 368, 452, 358, 371];
ind = [ 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34 ];
I=len(y)
J=max(ind)
assignment5_code = """
data {
    int<lower=0> I; #number of data
    int<lower=0> J; #number of participants
    int y[I];
    int ind[I];
}

parameters {
    real theta[J];
    real mu;
    real<lower=0> tau;
    real <lower=0>sigma;
}
model {
        sigma ~ uniform(0, 10000);
        tau ~ uniform(0, 10000);
        mu ~ uniform(-10000, 10000); 
        for (i in 1:I)
             y[i] ~ lognormal(theta[ind[i]],sigma);
        for (j in 1:J)
             theta[j] ~ normal(mu,tau);
}

generated quantities {
       
       real y_pred;
       real theta_pred;
       theta_pred = normal_rng(mu,tau);
       y_pred = normal_rng(theta_pred,sigma);
}



"""
assignment5_dat = {'I': I,'J':J,
               'y': y,
               'ind': ind}

model = pystan.StanModel(model_code=assignment5_code) #Create a model instance
fit = model.sampling(data=assignment5_dat,iter=100000,warmup=1000, chains=1) #Call the sampling using the model instance


#Extracting the data
parmExtract=fit.extract();

mu=parmExtract['mu'] #this gives the mu of log y
sigma=parmExtract['sigma']
theta=parmExtract['theta']
tau=parmExtract['tau']
logPosterior=parmExtract['lp__']
PredictedTheta=parmExtract['theta_pred']
PredictedReactionTime=parmExtract['y_pred']


#Reaction time for the dude

import pandas
    
def HDIofMCMC(SampleArray, credMass=.95): #Computes the HDI from an array of a unimodal samples of representative values.
    sortedarray=np.sort(SampleArray)
    CielingIndex=np.ceil(credMass*np.size(sortedarray))
    nCI=np.size(sortedarray)-CielingIndex
    ciWidth=np.zeros(int(nCI))
    for i in range(int(nCI)):
        ciWidth[i]=sortedarray[i+int(CielingIndex)]-sortedarray[i]
    HDImin=(sortedarray[np.argmin(ciWidth)])
    HDImax=(sortedarray[np.argmin(ciWidth)+int(CielingIndex)])
    HDIlim=np.array([HDImin,HDImax])
    return(HDIlim)
    
def mode(lst):
    d = {}
    for a in lst:
        if not a in d:
            d[a]=1
        else:
            d[a]+=1
    return [k for k,v in d.items() if v==max(d.values())]
    
TheDude= np.exp(theta[:,3]+sigma**2/2)

#posterior histogram for the dude
#plt.hist(TheDude,1000)



sns.distplot(TheDude, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2})
#posterior HDI for the dude
HDI=HDIofMCMC(TheDude,.95)
plt.plot(HDI,(0,0),'k',linewidth=6.0)
plt.text(np.mean(HDI), 0.0005, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(247, 0.00025, str(np.round(HDI[0],2)), fontsize=7,ha='center')
plt.text(417,0.00025, str(np.round(HDI[1],2)), fontsize=7,ha='center')

#calculate and print on plot, mean, median and mode
plt.text(500,0.009,'Mean ='+str(np.round(st.mean(TheDude),2)),va='top')
#plt.text(450,0.008,'Mode ='+str(np.round(st.mode(TheDude)],2)),va='bottom')
plt.text(500,0.007,'Median ='+str(np.round(st.median(TheDude),2)),va='bottom')

plt.title('The expected reaction time for the dude')
plt.xlabel('exp(θ4+σ2/2)')
plt.ylabel('Density')

#Reaction Time for the group
TheGroup = np.exp(mu+tau**2/2+sigma**2/2)

sns.distplot(TheGroup, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2})
#posterior HDI for the dude
HDI=HDIofMCMC(TheGroup,.95)
plt.plot(HDI,(0,0),'k',linewidth=6.0)
plt.text(np.mean(HDI), 0.0005, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(305, 0.001, str(np.round(HDI[0],2)), fontsize=7,ha='center')
plt.text(370,0.001, str(np.round(HDI[1],2)), fontsize=7,ha='center')

#calculate and print on plot, mean, median and mode
plt.text(400,0.023,'Mean ='+str(np.round(st.mean(TheGroup),2)),va='top')
#plt.text(450,0.008,'Mode ='+str(np.round(st.mode(TheDude),4)),va='bottom')
plt.text(400,0.020,'Median ='+str(np.round(st.median(TheGroup),2)),va='bottom')

plt.title('The expected reaction time for the group')
plt.xlabel('exp(μ+τ2/2+σ2/2)')
plt.ylabel('Density')

#Reaction time for a random individual
RandomIndividual = np.exp(PredictedReactionTime)

sns.distplot(RandomIndividual, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2})
#posterior HDI for the dude
HDI=HDIofMCMC(RandomIndividual,.95)
plt.plot(HDI,(0,0),'k',linewidth=6.0)
plt.text(np.mean(HDI), 0.0005, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(200, 0.00025, str(np.round(HDI[0],2)), fontsize=7,ha='center')
plt.text(550,0.00025, str(np.round(HDI[1],2)), fontsize=7,ha='center')

#calculate and print on plot, mean, median and mode
plt.text(1000,0.004,'Mean ='+str(np.round(st.mean(RandomIndividual),2)),va='top')
#plt.text(450,0.008,'Mode ='+str(np.round(st.mode(TheDude),4)),va='bottom')
plt.text(1000,0.0035,'Median ='+str(np.round(st.median(RandomIndividual),2)),va='bottom')

plt.title('The expected reaction time for a random individual')
plt.xlabel('exp(y_pred)')
plt.ylabel('Density')

#Comparing generated theta with the mean reaction times for each individual
thetaMean= np.zeros(np.max(ind))
index=0
n=0
labels= np.zeros(np.max(ind))
suming=0
indexing=0
for i in range(np.max(ind)):
    thetaMean[i]=sum(y[n:ind.count(i+1)+n])/ind.count(i+1)
    n=n+ind.count(i+1)
    labels[i]=indexing
    indexing=indexing+1
   
    

sns.boxplot(data=np.exp(theta), notch=False, meanline=True, showfliers=False)
plt.plot(labels,thetaMean,'*k')
plt.xlabel('Individuals',fontsize=20)
plt.ylabel(r'$\Theta$',fontsize=20)



