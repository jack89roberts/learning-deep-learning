#!/usr/bin/env python
# coding: utf-8

# # 03: Back Propagation
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# ## Matrix Notation

# In[ ]:





# ## Back Propagation

# In[ ]:





# ## Vanishing Gradients

# In[2]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def dsigmoid_dx(z):
    return sigmoid(z) * (1 - sigmoid(z))


z = np.linspace(-10, 10, 100)
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7, 3))
ax[0].plot(z, sigmoid(z), color="b", label=r"$\sigma(z)$")
ax[0].legend()
ax[0].set_xlabel("z")
ax[1].plot(z, dsigmoid_dx(z), color="orange", label=r"$\sigma'(z)$")
ax[1].legend()
ax[1].set_xlabel("z")


# ## ReLU

# In[3]:


def relu(z):
    r = z.copy()
    r[r < 0] = 0
    return r


def drelu_dz(z):
    dr = z.copy()
    dr[z < 0] = 0
    dr[z >= 0] = 1
    return dr


z = np.linspace(-5, 5, 200)
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7, 3))
ax[0].plot(z, relu(z), color="b", label=r"$\sigma(z)$")
ax[0].legend()
ax[0].set_xlabel("z")

ax[1].plot(z, drelu_dz(z), color="orange", label=r"$\sigma'(z)$")
ax[1].legend()
ax[1].set_xlabel("z")


# In[ ]:




