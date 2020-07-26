#!/usr/bin/env python
# coding: utf-8

# In[122]:


#checking the version of libraries
import sys
print('Python: {}'.format(sys.version))
import scipy
print('Scipy: {}'.format(sys.version))
import numpy
print('Numpy: {}'.format(sys.version))
import matplotlib
print('Matpplotlib: {}'.format(sys.version))
import pandas
print('Pandas: {}'.format(sys.version))
import sklearn
print('Sklearn: {}'.format(sys.version))


# In[123]:


import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[124]:


#loading the data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)


# In[125]:


#dimension of the dataset
print(dataset.shape)


# In[126]:


#take a peek at the data
print(dataset.head(20))


# In[127]:


#statistical summary
print(dataset.describe())


# In[128]:


#class distribution
print(dataset.groupby('class').size())


# In[129]:


# univariate plots - box and whisker plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# In[130]:


# histograms of the variable
dataset.hist()
pyplot.show()


# In[131]:


#multivariate plots
scatter_matrix(dataset)
pyplot.show()


# In[132]:


#creating a validation dataset
#splitting dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.2, random_state=1)


# In[136]:


#Logistic regression
#Linear Discriminant Analysis
#K-Nearest Neighbors
#Classification and Regression Trees
#Gaussian Naive Bayes
#Support Vector Machines

#buiding models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA',  LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append((('NB', GaussianNB())))
models.append(('SVM', SVC(gamma='auto')))


# In[137]:


#evaluate the created models
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[139]:


#compare our models
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparision')
pyplot.show()


# In[140]:


#make predicitons
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


# In[143]:


#evaluate our predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:




