#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Description: This program detects whether or not you have diabetes using ML


# In[16]:


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


# In[17]:


st.write("""
#Diabetes Detection
Detect if someone has diabetes using machine learning and python!
""")


# In[18]:


image=Image.open('diabetes.jpeg')
st.image(image,caption='ML',use_column_width=True)


# In[19]:


df=pd.read_csv('diabetes.csv')


# In[20]:


df.head()


# In[21]:


#Sub header
st.subheader('Data Information:')
st.dataframe(df)
st.write(df.describe())
chart=st.bar_chart(df)


# In[22]:


#Split the data into X and Y

X=df.iloc[:,0:8].values # first 8 coloumns
Y=df.iloc[:,-1].values # last column


# In[23]:


#split data into 75% training and 25% testing
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[24]:


def get_input():
    Pregnancies=st.sidebar.slider('pregnancies',0,17,3)
    Glucose=st.sidebar.slider('Glucose',0,199,117)
    BloodPressure=st.sidebar.slider('BloodPressure',0,122,172)
    SkinThickness=st.sidebar.slider('SkinThickness',0,99,23)
    Insulin=st.sidebar.slider('Insulin',0.0,846.0,30.0)
    BMI=st.sidebar.slider('BMI',0.0,67.1,32.0)
    DiabetesPedigreeFunction=st.sidebar.slider('DiabetesPedigreeFunction',0.078,2.42,0.3725)
    Age=st.sidebar.slider('Age',21,81,29)
    
    #dictionary
    user_data={'pregnancies':Pregnancies,
               'Glucose':Glucose,
               'BloodPressure':BloodPressure,
               'SkinThickness':SkinThickness,
               'Insulin':Insulin,
               'BMI':BMI,
               'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
               'Age':Age
               }
    
    
    #Transform the data into a data frame
    
    features=pd.DataFrame(user_data,index=[0])
    
    return features


# In[25]:


#Store input in a variable
user_input=get_input()


# In[26]:


#Display the user input
st.subheader('User Input:')
st.write(user_input)


# In[27]:


# Create and train the model
#Model-1
RandomForestClassifier= RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)


# In[32]:


#Show model metrics
st.subheader('RandomForestClassifier Test Accuracy Score:')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')


# In[33]:


prediction=RandomForestClassifier.predict(user_input)


# In[34]:


st.subheader('Classification ')
st.write(prediction)


# In[45]:


#Model-2
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB().fit(X_train,Y_train)


# In[71]:


#Show model metrics
st.subheader('Naive Bayes Test Accuracy Score:')
st.write(str(accuracy_score(Y_test,mnb.predict(X_test))*100)+'%')


# In[72]:


prediction1=mnb.predict(user_input)


# In[73]:


st.subheader('Classification naive Bayes ')
st.write(prediction1)


# In[74]:


#Model-3
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=1000)
lr.fit(X_train,Y_train)


# In[75]:


st.subheader('LogisticRegression Test Accuracy Score:')
st.write(str(accuracy_score(Y_test,lr.predict(X_test))*100)+'%')


# In[76]:


prediction2=lr.predict(user_input)


# In[77]:


st.subheader('Classification logistic Regression ')
st.write(prediction2)


# In[78]:


#Model -4
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm = 'brute', n_jobs=-1)
knn.fit(X_train,Y_train)


# In[79]:


st.subheader('KNeighborsClassifier Test Accuracy Score:')
st.write(str(accuracy_score(Y_test,knn.predict(X_test))*100)+'%')


# In[80]:


prediction3=knn.predict(user_input)


# In[81]:


st.subheader('Classification KNeighborsClassifier ')
st.write(prediction3)


# In[82]:


#Model-5
from sklearn.svm import LinearSVC
svm=LinearSVC(C=0.0001)
svm.fit(X_train,Y_train)


# In[84]:


st.subheader('LinearSVCTest Accuracy Score:')
st.write(str(accuracy_score(Y_test,svm.predict(X_test))*100)+'%')


# In[85]:


prediction4=svm.predict(user_input)


# In[86]:


st.subheader('Classification KNeighborsClassifier ')
st.write(prediction4)


# In[69]:


#Model -6
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,Y_train)


# In[87]:


st.subheader('DecisionTreeClassifier Accuracy Score:')
st.write(str(accuracy_score(Y_test,clf.predict(X_test))*100)+'%')


# In[88]:


prediction5=clf.predict(user_input)


# In[89]:


st.subheader('Classification DecisionTreeClassifier ')
st.write(prediction5)


# In[90]:


#Model-7
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
adb = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4),n_estimators=10,learning_rate=0.6)
adb.fit(X_train,Y_train)


# In[91]:


st.subheader('AdaBoostClassifier Accuracy Score:')
st.write(str(accuracy_score(Y_test,adb.predict(X_test))*100)+'%')


# In[92]:


prediction6=adb.predict(user_input)


# In[93]:


st.subheader('Classification AdaBoostClassifier ')
st.write(prediction6)


# In[ ]:




