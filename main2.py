import streamlit as st
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor

x = 4*np.random.rand(100)
y = np.sin(2*x + 1) + 0.1* np.random.randn(100)

knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(x.reshape(-1, 1), y)

