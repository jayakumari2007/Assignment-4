# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:45:10 2019

@author: jaykum
"""
import pandas



#Extracting the unlog values
#theta_individual = np.exp(theta+sigma**2/2) #individual
mu_adults = np.exp(mu+tau**2/2+sigma**2/2) #adults
phi_kids = np.exp(mu+phi+tau**2/2+sigma**2/2) #kids



# The HDI and mode functions 
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
    

    
#Task 1

#posterior histogram for kids

sns.distplot(phi, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2})
#posterior HDI for kids
HDI=HDIofMCMC(phi,.95)
plt.plot(HDI,(0,0),'k',linewidth=6.0)
ys = np.linspace(1, 3, 5)
plt.vlines(x=np.round(HDI[0],2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
plt.vlines(x=np.round(st.mean(phi),2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
plt.vlines(x=np.round(HDI[1],2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
plt.text(np.mean(HDI), 0.2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(0.23, 0.25, str(np.round(HDI[0],2)), fontsize=8,ha='center')
plt.text(0.50,0.25, str(np.round(HDI[1],2)), fontsize=8,ha='center')

#calculate and print on plot, mean
plt.text(0.38,4.65,str(np.round(st.mean(phi),2)),va='top')

plt.title('The expected log reaction time for kids')
plt.xlabel('phi')
plt.figure()
#posterior histogram for adults


sns.distplot(mu, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2})
#posterior HDI for adults
HDI=HDIofMCMC(mu,.95)
plt.plot(HDI,(0,0),'k',linewidth=6.0)
ys = np.linspace(1, 3, 10)
plt.vlines(x=np.round(HDI[0],2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
plt.vlines(x=np.round(st.mean(mu),2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
plt.vlines(x=np.round(HDI[1],2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
plt.text(np.mean(HDI), 0.2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(5.59, 0.25, str(np.round(HDI[0],2)), fontsize=8,ha='center')
plt.text(5.74,0.25, str(np.round(HDI[1],2)), fontsize=8,ha='center')

#calculate and print on plot, mean
plt.text(5.66,9,str(np.round(st.mean(mu),2)),va='top')

plt.title('The expected log reaction time for adults')
plt.xlabel('mu')

#Task 2

#plot tau for assignment 5

plt.subplot(1,2,1)
sns.distplot(tau5, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2})

HDI=HDIofMCMC(tau5,.95)
#plt.plot(HDI,(0,0),'k',linewidth=6.0)
ys = np.linspace(1, 8, 12)
plt.vlines(x=np.round(HDI[0],2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
plt.vlines(x=np.round(st.mean(tau5),2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
plt.vlines(x=np.round(HDI[1],2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
#plt.text(np.mean(HDI), 0.2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(np.round(HDI[0],2), 0.25, str(np.round(HDI[0],2)), fontsize=8,ha='center')
plt.text(np.round(HDI[1],2),0.25, str(np.round(HDI[1],2)), fontsize=8,ha='center')

#calculate and print on plot, mean
plt.text(np.round(st.mean(tau5),2),11,str(np.round(st.mean(tau5),2)),va='top')

plt.title('log tau for ass 5')
plt.xlabel('tau')

#plot tau for assignment 6
plt.subplot(1,2,2)
sns.distplot(tau, hist=True, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2})

HDI=HDIofMCMC(tau,.95)
#plt.plot(HDI,(0,0),'k',linewidth=6.0)
ys = np.linspace(1, 8, 14)
plt.vlines(x=np.round(HDI[0],2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
plt.vlines(x=np.round(st.mean(tau),2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
plt.vlines(x=np.round(HDI[1],2), ymin=0, ymax=len(ys), colors='k', linestyles='--', lw=1)
#plt.text(np.mean(HDI), 0.2, r'$95\%$ HDI', fontsize=10,ha='center')
plt.text(np.round(HDI[0],2), 0.25, str(np.round(HDI[0],2)), fontsize=8,ha='center')
plt.text(np.round(HDI[1],2),0.25, str(np.round(HDI[1],2)), fontsize=8,ha='center')

#calculate and print on plot, mean
plt.text(np.round(st.mean(tau),2),14,str(np.round(st.mean(tau),2)),va='top')

plt.title('log tau for ass 6')
plt.xlabel('tau')

#task 3

# prior for theta was

# theta[j] ~ normal(mu + phi*child_j[j],tau)



prior_kids= np.random.normal(st.mean(mu)+st.mean(phi), st.mean(tau), 10000)
prior_adults= np.random.normal(st.mean(mu), st.mean(tau), 10000)

prior_ass5 = np.random.normal(st.mean(mu5), st.mean(tau5), 10000)


sns.distplot(prior_kids,hist=False, kde=True, 
             bins=1000, color = 'darkblue', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2},label='kids prior')
plt.text(np.round(st.mean(prior_kids),2),1.75,str(np.round(st.mean(prior_kids),2)),va='top')
ys = np.linspace(0.25, 1, 2)
plt.vlines(x=np.round(st.mean(prior_kids),2), ymin=0, ymax=1.80, colors='b', linestyles='--', lw=1)
sns.distplot(prior_adults, hist=False, kde=True, 
             bins=1000, color = 'red', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2},label='adults prior')
plt.text(np.round(st.mean(prior_adults),2),1.85,str(np.round(st.mean(prior_adults),2)),va='top')
plt.vlines(x=np.round(st.mean(prior_adults),2), ymin=0, ymax=1.70, colors='r', linestyles='--', lw=1)
sns.distplot(prior_ass5, hist=False, kde=True, 
             bins=1000, color = 'green', 
             hist_kws={'edgecolor':'lightblue'},
             kde_kws={'linewidth': 2},label='ass5 prior')
plt.text(np.round(st.mean(prior_ass5),2),1.55,str(np.round(st.mean(prior_ass5),2)),va='top')
plt.vlines(x=np.round(st.mean(prior_ass5),2), ymin=0, ymax=1.45, colors='g', linestyles='--', lw=1)
plt.legend()

#task 4

N=10000
y_kids = np.zeros(N,)
y_adults = np.zeros(N,)
y_mixed = np.zeros(N,)
theta_kid = np.zeros(N,)
theta_adult = np.zeros(N,)
theta_mixed = np.zeros(N,)

indices = np.random.randint(0,mu.size,N)   #randomly chosen indices
kids_fraction = sum(child_j)/len(child_j)

for i,sample_i in enumerate(indices): 
    #knowing that it is a child
    theta_kid = np.random.randn(1)[0]*tau[sample_i] + mu[sample_i] + phi[sample_i]
    y_kids[i]=np.exp(np.random.randn(1)[0]*sigma[sample_i] + theta_kid )
    
    theta_adult = np.random.randn(1)[0]*tau[sample_i] + mu[sample_i] 
    y_adults[i]=np.exp(np.random.randn(1)[0]*sigma[sample_i] + theta_adult )
    
    #not knowing if it is a kid
    if np.random.rand()< kids_fraction:
       theta_mixed = np.random.randn(1)[0]*tau[sample_i] + mu[sample_i] + phi[sample_i]
       y_mixed[i]=np.exp(np.random.randn(1)[0]*sigma[sample_i] + theta_mixed ) 
    else:
       theta_mixed = np.random.randn(1)[0]*tau[sample_i] + mu[sample_i] 
       y_mixed[i]=np.exp(np.random.randn(1)[0]*sigma[sample_i] + theta_mixed ) 
                    
sns.distplot(y_adults,bins=100)
sns.distplot(y_kids,bins=100)
sns.distplot(y_mixed,bins=100)

adultsHDI = HDIofMCMC(y_adults,.95)
kidsHDI= HDIofMCMC(y_kids,.95)
mixedHDI = HDIofMCMC(y_mixed,.95)

adultsMean = st.mean(y_adults)
plt.text(np.round(adultsMean,2),0.005,str(np.round(adultsMean,2)),va='top')
kidsMean = st.mean(y_kids)
plt.text(np.round(kidsMean,2),0.003,str(np.round(kidsMean,2)),va='top')
mixedMean= st.mean(y_mixed)
plt.text(np.round(mixedMean,2),0.004,str(np.round(mixedMean,2)),va='top')

plt.title('treaction time for a random individual')
plt.xlabel(r'Reaction time, [ms]')
plt.ylabel('pdf')
plt.legend(['Adult','Child','Mixed'])
plt.show()

