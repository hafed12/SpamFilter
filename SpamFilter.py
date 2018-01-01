
# coding: utf-8

# In[1]:

from nltk import SnowballStemmer


# In[2]:

import numpy as np


# In[3]:

import pandas as pd


# In[4]:

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:

df= pd.read_csv('C:\\Users\\SONY\\python_projects\\smsspamcollection\\SMSSpamCollection.tsv', sep='\t', names=['status', 'message'])


# In[6]:

df.head()


# In[7]:

df.loc[df['status']=='ham','status']=1


# In[8]:

df.loc[df['status']=='spam','status']=0


# In[9]:

df.head()


# In[10]:

len(df)


# In[11]:

df_x=df['message']


# In[12]:

df_y=df['status']


# In[13]:

stemmer=SnowballStemmer('english')


# In[14]:

for word in df_x:
    stemmer.stem(word)


# In[15]:

transformer= TfidfVectorizer(min_df=1,stop_words='english')


# In[16]:

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


# In[17]:

x_train_tfidf = transformer.fit_transform(x_train)


# In[18]:

a=x_train_tfidf.toarray()


# In[19]:

len(x_train)


# In[20]:

transformer.inverse_transform(a[0])


# In[21]:

mnb= MultinomialNB()


# In[22]:

y_train=y_train.astype('int')


# In[23]:

mnb.fit(x_train_tfidf,y_train)


# In[24]:

x_test_tfidf = transformer.transform(x_test)


# In[25]:

pred1 =mnb.predict(x_test_tfidf)


# In[26]:

pred1


# In[27]:

result1 = np.array(y_test)


# In[28]:

result1


# In[29]:

count1=0


# In[30]:

for i in range(len(pred1)):
    if result1[i]==pred1[i]:
        count1 = count1+1


# In[31]:

count1


# In[32]:

len(pred1)


# In[33]:

(count1/len(pred1))*100


# In[34]:

from sklearn.svm import SVC, NuSVC, LinearSVC


# In[35]:

svm = LinearSVC()


# In[36]:

svm.fit(x_train_tfidf,y_train)


# In[37]:

pred2=svm.predict(x_test_tfidf)


# In[38]:

result2 = np.array(y_test)


# In[39]:

result2


# In[40]:

count2=0


# In[41]:

for i in range(len(pred2)):
    if pred2[i]==result2[i]:
        count2 = count2+1


# In[42]:

count2


# In[43]:

len(pred2)


# In[44]:

(count2/len(pred2))*100


# In[ ]:




# In[ ]:



