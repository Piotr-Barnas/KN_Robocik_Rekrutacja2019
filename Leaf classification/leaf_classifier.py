#!/usr/bin/env python
# coding: utf-8

# # Leaf classifier

# In[198]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[199]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[200]:


train.head(2)


# In[201]:


test.head(2)


# In[202]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss


# In[203]:


x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

x_test = test.drop('id', axis=1)


# In[204]:


#wyznaczam średnią i odchylenie standardowe zbioru x_train, które zostanie użyte do standaryzacji zbiorów x_train i x_test
scaler = StandardScaler().fit(x_train)


# In[205]:


#standaryzuję
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# #### Zbiór treningowy dodatkowo podzieliłem na kolejne dwa podzbiory train/test (w peoporcji 80:20) żeby sprawdzić czy zaimplementowany model dobrze klasyfikuje gatunki drzew

# In[206]:


#w zbiorze treningowym jest 990 wierszy danych przy czym mamy 99 gatunków, dlatego użyłem 10 podziałów 
strati = StratifiedShuffleSplit(10, 0.2)
for train_index, test_index in strati.split(x_train, y_train):
    X_train, X_test = x_train[train_index], x_train[test_index]
    Y_train, Y_test = y_train[train_index], y_train[test_index]


# In[207]:


classifier = KNeighborsClassifier(n_neighbors=1)


# In[208]:


classifier.fit(X_train, Y_train)


# In[209]:


prediction = classifier.predict(X_test)


# #### Trafność predykcji wypadła na poziomie 98.48%

# In[210]:


accuracy_score(Y_test, prediction)


# ####  Sprawdzam, dla jakiej wartości k model ma największą dokładonść. Na wykresie średniej liczby błednych predykcji w zależnosci od parametru k widać, że wraz ze wzrostem k rośnie również błąd, dlatego w modelu wykorzystał k=1

# In[211]:


error_rate = []

for i in range(1,40):
    
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train,Y_train)
    prediction_i = classifier.predict(X_test)
    error_rate.append(np.mean(prediction_i != Y_test))


# In[212]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# #### Poprzedni model dopasowywałem do 80% danych treningowych, dlatego teraz dopasowuję jeszcze raz do całego zbioru

# In[213]:


#ostateczny model
better_classifier = KNeighborsClassifier(n_neighbors=1)
better_classifier.fit(x_train, y_train)
y_prob = better_classifier.predict_proba(x_test)

#nazwy kolumn i id wierszy do dataframe'a z wynikami
species = le.classes_
ids = test['id']


# In[214]:


sub = pd.DataFrame(y_prob, index=ids, columns=species)


# In[216]:


sub.to_csv('submission.csv')

