# input library
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# define x and y
x = 4*np.random.rand(100)
y = np.sin(2*x+1) + 0.1*np.random.randn(100)

# create side bar for select the classifier
st.sidebar.title('Classifier Selection')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Decision Tree', 'Random Forest', 'Neural Network'))

# giving the k-value as a slide bar
k = st.sidebar.slider('K Value', 1, 20, 1)

# check condition if select the side bar to calculate the accuracy
if classifier == 'KNN':
  knn = KNeighborsRegressor(n_neighbors=5)
  knn.fit(x.reshape(-1, 1), y)
  y_pred = knn.predict(x.reshape(-1, 1))
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.scatter(x, y_pred)
  st.pyplot(fig)
  
if classifier == 'SVM':
  svm = SVR(kernel='rbf')
  svm.fit(x.reshape(-1,1), y)
  y_pred = svm.predict(x.reshape(-1,1))
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.scatter(x, y_pred)
  st.pyplot(fig)
  
if classifier == 'Neural Network':
  nn = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=1000)
  nn.fit(x.reshape(-1,1), y)
  y_pred = nn.predict(x.reshape(-1,1))
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.scatter(x, y_pred)
  st.pyplot(fig)
  
if classifier == 'Decision Tree':
  dt = DecisionTreeRegressor()
  dt.fit(x.reshape(-1,1), y)
  y_pred = dt.predict(x.reshape(-1,1))
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.scatter(x, y_pred)
  st.pyplot(fig)
  
if classifier == 'Random Forest':
  rf = RandomForestRegressor(n_estimators=100)
  rf.fit(x.reshape(-1,1), y)
  y_pred = rf.predict(x.reshape(-1,1))
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.scatter(x, y_pred)
  st.pyplot(fig)
