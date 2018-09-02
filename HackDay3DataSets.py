
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from pandas import Series,DataFrame
# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

#Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#creating a data frame
from sklearn.model_selection import train_test_split
performance=pd.read_csv('C:/Users/Harsh Naicker/Desktop/xAPI-Edu-Data.csv')
performance=performance.drop(['gender'], axis=1).drop(['NationalITy'], axis=1).drop(['PlaceofBirth'], axis=1).drop(['StageID'], axis=1).drop(['GradeID'], axis=1).drop(['SectionID'], axis=1).drop(['Topic'], axis=1).drop(['Semester'], axis=1).drop(['Relation'], axis=1).drop(['AnnouncementsView'], axis=1).drop(['ParentschoolSatisfaction'], axis=1)

data=performance

#print(performance.head(5))
#print(performance_train_df)
#Converting non numeric data variables to numeric data variables
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()
label.fit(performance['Class'])
performance['Class']=label.transform(performance['Class'])

#print(performance['Class'])
label=LabelEncoder()
label.fit(performance['ParentAnsweringSurvey'])
performance['ParentAnsweringSurvey']=label.transform(performance['ParentAnsweringSurvey'])

label=LabelEncoder()
label.fit(performance['StudentAbsenceDays'])
performance['StudentAbsenceDays']=label.transform(performance['StudentAbsenceDays'])

x_train, x_test, y_train, y_test=train_test_split(performance.drop(['Class'], axis=1), performance['Class'], test_size=0.4, random_state=1)
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

#import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#create a gaussian classifier
model = GaussianNB()

#train the model using the training data sets
model.fit(x_train, y_train)
#predict output
predicted=model.predict(x_test)


# In[2]:


data.raisedhands.shape,predicted.shape


# In[3]:


plt.scatter(data.raisedhands[0:192],predicted,c=data.Class[0:192])
plt.xlabel('RaisedHands')
plt.ylabel('Y_pred')
plt.xlim(1,20)
#plt.legend(loc='upper left')


# In[4]:


data.VisITedResources.shape,predicted.shape
plt.scatter(data.VisITedResources[0:192],predicted,c=data.Class[0:192])
plt.xlabel('VisitedResources')
plt.ylabel('Y_pred')
plt.xlim(1,20)


# In[5]:


data.Discussion.shape,predicted.shape
plt.scatter(data.Discussion[0:192],predicted,c=data.Class[0:192])
plt.xlabel('Discussion')
plt.ylabel('Y_pred')
plt.xlim(1,20)


# In[6]:


data.ParentAnsweringSurvey.shape,predicted.shape
plt.scatter(data.ParentAnsweringSurvey[0:192],predicted,c=data.Class[0:192])
plt.xlabel('ParentAnsweringSurvey')
plt.ylabel('Y_pred')
plt.xlim(-2,2)


# In[7]:


data.StudentAbsenceDays.shape,predicted.shape
plt.scatter(data.StudentAbsenceDays[0:192],predicted,c=data.Class[0:192])
plt.xlabel('StudentAbsenceDays')
plt.ylabel('Y_pred')
plt.xlim(-2,2)


# In[8]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(predicted, y_test)
print(cm)
efficiency=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100
print("Efficiency of prediction model : ",efficiency,"%")


# In[22]:


performance=pd.read_csv('C:/Users/Harsh Naicker/Downloads/DB2.csv')
performance=performance.drop(['gender'], axis=1).drop(['NationalITy'], axis=1).drop(['PlaceofBirth'], axis=1).drop(['StageID'], axis=1).drop(['GradeID'], axis=1).drop(['SectionID'], axis=1).drop(['Topic'], axis=1).drop(['Semester'], axis=1).drop(['Relation'], axis=1).drop(['AnnouncementsView'], axis=1).drop(['ParentschoolSatisfaction'], axis=1)

data=performance
label=LabelEncoder()
label.fit(performance['Class'])
performance['Class']=label.transform(performance['Class'])
label=LabelEncoder()
label.fit(performance['ParentAnsweringSurvey'])
performance['ParentAnsweringSurvey']=label.transform(performance['ParentAnsweringSurvey'])

label=LabelEncoder()
label.fit(performance['StudentAbsenceDays'])
performance['StudentAbsenceDays']=label.transform(performance['StudentAbsenceDays'])

x_train, x_test, y_train, y_test=train_test_split(performance.drop(['Class'], axis=1), performance['Class'], test_size=0.4, random_state=12)
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
model = GaussianNB()

#train the model using the training data sets
model.fit(x_train, y_train)
#predict output
predicted=model.predict(x_test)


# In[10]:


data.raisedhands.shape,predicted.shape


# In[13]:


plt.scatter(data.raisedhands[0:192],predicted,c=data.Class[0:192])
plt.xlabel('RaisedHands')
plt.ylabel('Y_pred')
plt.xlim(1,20)


# In[14]:


data.VisITedResources.shape,predicted.shape
plt.scatter(data.VisITedResources[0:192],predicted,c=data.Class[0:192])
plt.xlabel('VisitedResources')
plt.ylabel('Y_pred')
plt.xlim(1,20)


# In[15]:


data.Discussion.shape,predicted.shape
plt.scatter(data.Discussion[0:192],predicted,c=data.Class[0:192])
plt.xlabel('Discussion')
plt.ylabel('Y_pred')
plt.xlim(1,20)


# In[16]:


data.ParentAnsweringSurvey.shape,predicted.shape
plt.scatter(data.ParentAnsweringSurvey[0:192],predicted,c=data.Class[0:192])
plt.xlabel('ParentAnsweringSurvey')
plt.ylabel('Y_pred')
plt.xlim(-2,2)


# In[17]:


data.StudentAbsenceDays.shape,predicted.shape
plt.scatter(data.StudentAbsenceDays[0:192],predicted,c=data.Class[0:192])
plt.xlabel('StudentAbsenceDays')
plt.ylabel('Y_pred')
plt.xlim(-2,2)


# In[18]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(predicted, y_test)
print(cm)
efficiency=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100
print("Efficiency of prediction model : ",efficiency,"%")


# In[20]:





# In[21]:




