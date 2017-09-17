
# coding: utf-8

# In[6]:

get_ipython().magic(u'matplotlib inline')
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
figsize(11, 9)
import scipy.stats as stats


# In[9]:

dist = stats.beta
n_trials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 50, 500]
data = stats.bernoulli.rvs(0.5, size = n_trials[-1])
x = np.linspace(0, 1, 100)


# In[10]:

for k, N in enumerate(n_trials):                   # k the key, N the value for enumerate(n_trials) 
    sx = plt.subplot(len(n_trials) / 2, 2, k + 1)  # define rows and columns for subplot
    plt.xlabel("$p$, probability of heads")         if k in [0, len(n_trials) - 1] else None   # $character string$ for italics
    plt.setp(sx.get_yticklabels(), visible=False)
    heads = data[:N].sum()
    y = dist.pdf(x, 1 + heads, 1 + N - heads)      # recall dist.pdf prob density function of beta distribution 
    plt.plot(x, y, label = "observe %d tosses,\n %d heads" % (N, heads))
    plt.fill_between(x, 0, y, color = "#348ABD", alpha = 0.4)
    plt.vlines(0.5, 0, 4, color = "k", linestyles="--", lw = 1)

    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight = True)


# In[35]:

n = 100  # number of tosses
h = 61


a, b = 10, 10  # parameters of the beta distrubution 
prior = st.beta(a, b)
post = st.beta(h + a, n - h + b)
ci = post.interval(0.95)

