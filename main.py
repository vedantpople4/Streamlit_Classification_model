import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


st.title("Streamlit Classification Models")


dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
classifier_name = st.sidebar.selectbox("Select classifier", ("KNN", "SVM", "Random Forest "))


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else: 
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x,y

x,y = get_dataset(dataset_name)

st.write("Shape of Selected Dataset is: ",x.shape)
st.write("Number of Classes are: ", len(np.unique(y)))

def add_parameter_uci(clf_name):
    params = {}
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C",0.1,10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_uci(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
        
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
        
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"],random_state=42)
        
    return clf

clf = get_classifier(classifier_name, params)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)

clf.fit(x_train,y_train)
pred = clf.predict(x_test)

acc = accuracy_score(y_test, pred)
st.write(f"Classifier selected is : {classifier_name}")
st.write(f"Accuracy of model is : {acc}")

pca = PCA(2)
x_pca = pca.fit_transform(x)

x1 = x_pca[:,0]
x2 = x_pca[:,1]

fig = plt.figure()
plt.scatter(x1, x2, alpha=0.8, cmap="virdis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)








    