#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pickle
import numpy as np
import operator
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import math


# In[25]:


with open('train.pkl', 'rb') as f:
    data = pickle.load(f)
with open('test.pkl', 'rb') as f:
    data1 = pickle.load(f)

x_train=data[:,:-1]
y_train=data[:,1]
x_test=data1[:,:-1]
y_test=data1[:,1]

x_cords_datasets=np.array(np.array_split(x_train,10))
y_cords_datasets=np.array(np.array_split(y_train,10))

final_var=[]
final_bias=[]
final_bias_square=[]
final_error=[]
final_irrerror=[]


# In[26]:



for degree in range(1,21):
    temp=[]
    for ds in range(10):
        poly_here=PolynomialFeatures(degree=degree, include_bias=False)
        xtr=poly_here.fit_transform(x_cords_datasets[ds])
        xts=poly_here.fit_transform(x_test)
        reg=LinearRegression().fit(xtr,y_cords_datasets[ds])
        temp.append(reg.predict(xts))
        
    arr_error=np.array((y_test - temp)**2)
    final_error.append(np.mean(arr_error))#mse
    
    arr_bias=np.array((np.mean(temp,axis=0)-y_test))
    final_bias.append(abs(np.mean(arr_bias))) #bias
    
    arr_bias_square=np.array(np.mean(temp,axis=0)-y_test)**2
    final_bias_square.append(np.mean(arr_bias_square)) #bias2
    
    arr_var=np.array(np.var(temp,axis=0))
    final_var.append(np.mean(arr_var)) #variance
    final_irrerror.append(final_error[degree-1]-(final_bias_square[degree-1]+final_var[degree-1]))

    ##to print the values in table
arr=[i for i in range(1,21)]
print("        Bias          "," bias square       ",'       ',"Variance     ")
for i in range(20):    
    print(i+1,' ',final_bias[i],' ',final_bias_square[i],' ',final_var[i])


plt.title('Bias^2 vs Variance')
plt.plot(arr, final_bias_square, color="b")
plt.plot(arr, final_var, color="orange") 
plt.plot(arr,final_error,color="black")
plt.plot(arr,final_irrerror,color="green")
# plt.plot(arr, final_error, color="pink")
plt.xlabel("Model Complexity")
plt.ylabel(" Irreducible Error")
plt.legend(('Bias Squared', 'Variance','MSE','Irreducible Error'), loc='best')
plt.show()

arr1=[i for i in range(1,21)]
print( "Mean square error",  "   IrreducibleError ")
for i in range(20):    
     print(i+1,' ',final_error[i],' ',final_irrerror[i])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




