#!/usr/bin/env python
# coding: utf-8

# # Anomaly detection

# In[39]:


import numpy as np
# Read text file and assign data to an array
df_array = np.loadtxt("anomaly_detection.txt")


# In[40]:


df_array


# In[41]:


#DEfine anomaly detection fucntion
def anomaly_detection(x):
    import math
    found = True
    while found: 
        for i in range(len(x)): 
            mean = (np.mean(x)*len(x)-x[i])/(len(x)-1)
            std = math.sqrt((np.std(x)**2*len(x)-(x[i]-mean)**2)/(len(x)-1))
            distance = abs((x[i]-mean)/std)
            if distance > 3:
                print("Remove {:6.2f} from the list because it's {:3.2f} times of standard deviation of the list without it.".format(x[i],distance)) 
                print("{:6.2f} is removed from the list!\n".format(x[i]))
                x = np.delete(x,i)
                found=True
                break
            found = False

    print("No more anomaly is detected!")


# In[42]:


#Calling anomaly detection function for the array we got from the dataset
anomaly_detection(df_array)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




