import streamlit as st
import seaborn as sns

df = sns.load_dataset('iris')
df

x = df.iloc[:, :-1]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

st.sidebar.title('Classification')
classification = st.sidebar.selectbox('Select classification', ['KNN', 'SVM'])
if classifier == 'KNN':
  knn = KNeighborsClassifier(n_neighbors=3)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test)
  accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'SVM':
  svm = SVC()
  svm.fit(x_train, y_train)
  y_pred = svm.predict(x_test)
  accuracy_score(y_test, y_pred)
  st.write(acc)
