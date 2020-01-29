# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:11:44 2020

@author: jaykum
"""

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

#input data
y = [607, 583, 521, 494, 369, 782, 570, 678, 467, 620, 425, 395, 346, 361, 310,

300, 382, 294, 315, 323, 421, 339, 398, 328, 335, 291, 329, 310, 294, 321, 286,

349, 279, 268, 293, 310, 259, 241, 243, 272, 247, 275, 220, 245, 268, 357, 273,

301, 322, 276, 401, 368, 149, 507, 411, 362, 358, 355, 362, 324, 332, 268, 259,

274, 248, 254, 242, 286, 276, 237, 259, 251, 239, 247, 260, 237, 206, 242, 361,

267, 245, 331, 357, 284, 263, 244, 317, 225, 254, 253, 251, 314, 239, 248, 250,

200, 256, 233, 427, 391, 331, 395, 337, 392, 352, 381, 330, 368, 381, 316, 335,

316, 302, 375, 361, 330, 351, 186, 221, 278, 244, 218, 126, 269, 238, 194, 384,

154, 555, 387, 317, 365, 357, 390, 320, 316, 297, 354, 266, 279, 327, 285, 258,

267, 226, 237, 264, 510, 490, 458, 425, 522, 927, 555, 550, 516, 548, 560, 545,

633, 496, 498, 223, 222, 309, 244, 207, 258, 255, 281, 258, 226, 257, 263, 266,

238, 249, 340, 247, 216, 241, 239, 226, 273, 235, 251, 290, 473, 416, 451, 475,

406, 349, 401, 334, 446, 401, 252, 266, 210, 228, 250, 265, 236, 289, 244, 327,

274, 223, 327, 307, 338, 345, 381, 369, 445, 296, 303, 326, 321, 309, 307, 319,

288, 299, 284, 278, 310, 282, 275, 372, 295, 306, 303, 285, 316, 294, 284, 324,

264, 278, 369, 254, 306, 237, 439, 287, 285, 261, 299, 311, 265, 292, 282, 271,

268, 270, 259, 269, 249, 261, 425, 291, 291, 441, 222, 347, 244, 232, 272, 264,

190, 219, 317, 232, 256, 185, 210, 213, 202, 226, 250, 238, 252, 233, 221, 220,

287, 267, 264, 273, 304, 294, 236, 200, 219, 276, 287, 365, 438, 420, 396, 359,

405, 397, 383, 360, 387, 429, 358, 459, 371, 368, 452, 358, 371]

#index of each data
ind = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5,

5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8,

 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,

 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,

  11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,

   12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14,

   14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16,

   16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20,

   20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21,

   21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,

   22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24,

   24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25,

   25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28,

   28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30,

   30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 34, 34, 34, 34, 34,

   34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34];
       
#indicator if the individual is a child or not, 1-child, 0-adult
child_j = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1,

           0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0];


    
#attempt number of each observation
x = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2, 3,

4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3,

4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,

15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,

20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2,

3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4,

5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5,

6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8,

9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1,

2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4,

5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7,

8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4,

5, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
     

     
I=len(y)
J=max(ind)
assignment7_code = """
data {
    int<lower=0> I; #number of data
    int<lower=0> J; #number of participants
    int y[I];
    int ind[I];
    int child_j[J];
    int x[I];
}

parameters {
    real theta0[J]; #intercept for each individual
    real theta1[J]; #slope for each individual 
    real <lower=0>sigma; #variation around regression line (same for all)
    real mu0; #the mean for the group intercept
    real<lower=0> tau0; #the std dev for the group intercept
    real mu1; #the mean for the group slope
    real<lower=0> tau1; # the standard deviation for the group slope
    real phi0; #term added to the intercept if it is a kid
    real phi1; #term added to the slope if it is a kid
}
model {
        sigma ~ uniform(0, 10000);
        tau0 ~ uniform(0, 10000);
        tau1 ~ uniform(0, 10000);
        mu0 ~ uniform(-10000, 10000); 
        mu1 ~ uniform(-10000, 10000); 
        phi0 ~ uniform(-10000, 10000); 
        phi1 ~ uniform(-10000, 10000); 
        for (i in 1:I)
             y[i] ~ lognormal(theta0[ind[i]] + theta1[ind[i]]*x[i],sigma);
        for (j in 1:J)
             theta0[j] ~ normal(mu0 + phi0*child_j[j], tau0);
        for (j in 1:J) 
             theta1[j] ~ normal(mu1 + phi1*child_j[j], tau1);
}

generated quantities {
       
       
       
}



"""
assignment7_dat = {'I': I,'J':J,
               'y': y,
               'ind': ind,'child_j':child_j,'x':x}

model = pystan.StanModel(model_code=assignment7_code) #Create a model instance
control = {}
control['max_treedepth'] = 20
control['adapt_delta'] = 0.99

fit = model.sampling(data=assignment7_dat,iter=100000,warmup=1000, chains=1) #Call the sampling using the model instance

#Extracting the data
parmExtract=fit.extract();

mu0=parmExtract['mu0']
mu1=parmExtract['mu1'] #this gives the mu of log y
sigma=parmExtract['sigma']
theta0=parmExtract['theta0']
theta1=parmExtract['theta1']
tau0=parmExtract['tau0']
tau1=parmExtract['tau1']
phi0=parmExtract['phi0']
phi1=parmExtract['phi1']
logPosterior=parmExtract['lp__']


#plots

Theta1_1_Real=np.exp(theta1[:,0]) # slope for participant 1
Theta1_3_Real=np.exp(theta1[:,2]) #slope for participant 3
Theta1_4_Real=np.exp(theta1[:,3]) #slope for participant 4

fig, axes = plt.subplots(nrows=1, ncols=3)
fig.tight_layout()

#plot for participant 1
plt.subplot(131)
sns.distplot(Theta1_1_Real, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 1})

HDI=HDIofMCMC(Theta1_1_Real,.95)
plt.plot(HDI,(0,0),'darkblue',linewidth=8.0)
ys = np.linspace(1, 3, 5)
plt.vlines(x=np.round(HDI[0],30), ymin=0, ymax=8, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(st.mean(Theta1_1_Real),2), ymin=0, ymax=45, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(HDI[1],30), ymin=0, ymax=8, colors='darkblue', linestyles='--', lw=2)
#plt.text(np.mean(HDI), 2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(HDI[0], 8, str(np.round(HDI[0],2)), fontsize=8,ha='right',va='bottom')
plt.text(HDI[1], 8, str(np.round(HDI[1],2)), fontsize=8,ha='left',va='bottom')
plt.text(st.mean(HDI),50,str(np.round(st.mean(HDI),2)),va='top')
plt.xlabel('exp⁡(θ1[1])')


plt.subplot(132)
sns.distplot(Theta1_3_Real, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2})

HDI=HDIofMCMC(Theta1_3_Real,.95)
plt.plot(HDI,(0,0),'darkblue',linewidth=8.0)
ys = np.linspace(1, 3, 5)
plt.vlines(x=np.round(HDI[0],30), ymin=0, ymax=8, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(st.mean(Theta1_3_Real),2), ymin=0, ymax=90, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(HDI[1],30), ymin=0, ymax=8, colors='darkblue', linestyles='--', lw=2)
#plt.text(np.mean(HDI), 2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(HDI[0], 8, str(np.round(HDI[0],2)), fontsize=8,ha='right',va='bottom')
plt.text(HDI[1], 8, str(np.round(HDI[1],2)), fontsize=8,ha='left',va='bottom')
plt.text(st.mean(HDI),90,str(np.round(st.mean(HDI),2)),va='top')
plt.xlabel('exp⁡(θ1[3])')


plt.subplot(133)
sns.distplot(Theta1_4_Real, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2})

HDI=HDIofMCMC(Theta1_4_Real,.95)
plt.plot(HDI,(0,0),'darkblue',linewidth=8.0)
ys = np.linspace(1, 3, 5)
plt.vlines(x=np.round(HDI[0],30), ymin=0, ymax=8, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(st.mean(Theta1_4_Real),2), ymin=0, ymax=85, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(HDI[1],30), ymin=0, ymax=8, colors='darkblue', linestyles='--', lw=2)
#plt.text(np.mean(HDI), 2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(HDI[0], 8, str(np.round(HDI[0],2)), fontsize=8,ha='right',va='bottom')
plt.text(HDI[1], 8, str(np.round(HDI[1],2)), fontsize=8,ha='left',va='bottom')
plt.text(st.mean(HDI),85,str(np.round(st.mean(HDI),2)),va='top')
plt.xlabel('exp⁡(θ1[4])')


#plots

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:07:47 2020

@author: jaykum
"""



fig, axes = plt.subplots(nrows=1, ncols=3)
fig.tight_layout()

#plot for participant 1
plt.subplot(131)
sns.distplot(No1Response1, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 1}, label='x=1')

HDI=HDIofMCMC(No1Response1,.95)
plt.plot(HDI,(0,0),'darkblue',linewidth=8.0)
ys = np.linspace(1, 3, 5)
plt.vlines(x=np.round(HDI[0],3), ymin=0, ymax=0.002, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(st.mean(No1Response1),2), ymin=0, ymax=0.011, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(HDI[1],3), ymin=0, ymax=0.002, colors='darkblue', linestyles='--', lw=2)
#plt.text(np.mean(HDI), 2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(HDI[0], 0.002, str(np.round(HDI[0],2)), fontsize=8,ha='right',va='bottom')
plt.text(HDI[1], 0.002, str(np.round(HDI[1],2)), fontsize=8,ha='left',va='bottom')
plt.text(st.mean(HDI),0.011,str(np.round(st.mean(HDI),2)),va='top')

sns.distplot(No1Response5, hist=True, kde=True, 
             bins=1000, color = 'darkgreen', 
             hist_kws={'edgecolor':'lightgreen'},
             kde_kws={'linewidth': 1},label='x=2')

HDI=HDIofMCMC(No1Response5,.95)
plt.plot(HDI,(0,0),'g',linewidth=8.0)
ys = np.linspace(1, 3, 5)
plt.vlines(x=np.round(HDI[0],3), ymin=0, ymax=0.002, colors='k', linestyles='--', lw=2)
plt.vlines(x=np.round(st.mean(No1Response5),2), ymin=0, ymax=0.013, colors='k', linestyles='--', lw=2)
plt.vlines(x=np.round(HDI[1],3), ymin=0, ymax=0.002, colors='k', linestyles='--', lw=2)
#plt.text(np.mean(HDI), 2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(HDI[0], 0.002, str(np.round(HDI[0],2)), fontsize=8,ha='right',va='bottom')
plt.text(HDI[1], 0.002, str(np.round(HDI[1],2)), fontsize=8,ha='left',va='bottom')
plt.text(st.mean(HDI),0.013,str(np.round(st.mean(HDI),2)),va='top')
plt.legend(labels=['x=1', 'x=5'])
plt.xlabel('exp⁡(θ0[1]+θ1[1]x+σ^2/2])')

#plot for participant 3
plt.subplot(132)
sns.distplot(No3Response1, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 1})

HDI=HDIofMCMC(No3Response1,.95)
plt.plot(HDI,(0,0),'darkblue',linewidth=8.0)
ys = np.linspace(1, 3, 5)
plt.vlines(x=np.round(HDI[0],3), ymin=0, ymax=0.003, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(st.mean(No3Response1),2), ymin=0, ymax=0.021, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(HDI[1],3), ymin=0, ymax=0.003, colors='darkblue', linestyles='--', lw=2)
#plt.text(np.mean(HDI), 2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(HDI[0], 0.003, str(np.round(HDI[0],2)), fontsize=8,ha='right',va='bottom')
plt.text(HDI[1], 0.003, str(np.round(HDI[1],2)), fontsize=8,ha='left',va='bottom')
plt.text(st.mean(HDI),0.021,str(np.round(st.mean(HDI),2)),va='top')

sns.distplot(No3Response5, hist=True, kde=True, 
             bins=1000, color = 'darkgreen', 
             hist_kws={'edgecolor':'lightgreen'},
             kde_kws={'linewidth': 1})

HDI=HDIofMCMC(No3Response5,.95)
plt.plot(HDI,(0,0),'g',linewidth=8.0)
ys = np.linspace(1, 3, 5)
plt.vlines(x=np.round(HDI[0],3), ymin=0, ymax=0.003, colors='k', linestyles='--', lw=2)
plt.vlines(x=np.round(st.mean(No3Response5),2), ymin=0, ymax=0.025, colors='k', linestyles='--', lw=2)
plt.vlines(x=np.round(HDI[1],3), ymin=0, ymax=0.003, colors='k', linestyles='--', lw=2)
#plt.text(np.mean(HDI), 2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(HDI[0], 0.003, str(np.round(HDI[0],2)), fontsize=8,ha='right',va='bottom')
plt.text(HDI[1], 0.003, str(np.round(HDI[1],2)), fontsize=8,ha='left',va='bottom')
plt.text(st.mean(HDI),0.025,str(np.round(st.mean(HDI),2)),va='top')
plt.legend(labels=['x=1', 'x=5'])
plt.xlabel('exp⁡(θ0[3]+θ1[3]x+σ^2/2])')

#plot for participant 4
plt.subplot(133)
sns.distplot(No4Response1, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 1})

HDI=HDIofMCMC(No4Response1,.95)
plt.plot(HDI,(0,0),'darkblue',linewidth=8.0)
ys = np.linspace(1, 3, 5)
plt.vlines(x=np.round(HDI[0],3), ymin=0, ymax=0.002, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(st.mean(No4Response1),2), ymin=0, ymax=0.011, colors='darkblue', linestyles='--', lw=2)
plt.vlines(x=np.round(HDI[1],3), ymin=0, ymax=0.002, colors='darkblue', linestyles='--', lw=2)
#plt.text(np.mean(HDI), 2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(HDI[0], 0.002, str(np.round(HDI[0],2)), fontsize=8,ha='right',va='bottom')
plt.text(HDI[1], 0.002, str(np.round(HDI[1],2)), fontsize=8,ha='left',va='bottom')
plt.text(st.mean(HDI),0.010,str(np.round(st.mean(HDI),2)),va='top')

sns.distplot(No4Response5, hist=True, kde=True, 
             bins=1000, color = 'darkgreen', 
             hist_kws={'edgecolor':'lightgreen'},
             kde_kws={'linewidth': 1})

HDI=HDIofMCMC(No4Response5,.95)
plt.plot(HDI,(0,0),'g',linewidth=8.0)
ys = np.linspace(1, 3, 5)
plt.vlines(x=np.round(HDI[0],3), ymin=0, ymax=0.002, colors='k', linestyles='--', lw=2)
plt.vlines(x=np.round(st.mean(No4Response5),2), ymin=0, ymax=0.011, colors='k', linestyles='--', lw=2)
plt.vlines(x=np.round(HDI[1],3), ymin=0, ymax=0.002, colors='k', linestyles='--', lw=2)
#plt.text(np.mean(HDI), 2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(HDI[0], 0.002, str(np.round(HDI[0],2)), fontsize=8,ha='right',va='bottom')
plt.text(HDI[1], 0.002, str(np.round(HDI[1],2)), fontsize=8,ha='left',va='bottom')
plt.text(st.mean(HDI),0.010,str(np.round(st.mean(HDI),2)),va='top')
plt.legend(labels=['x=1', 'x=5'])
plt.xlabel('exp⁡(θ0[4]+θ1[4]x+σ^2/2])')






    