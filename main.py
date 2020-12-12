import streamlit as st
import numpy as np
from sklearn import datasets

st.title("Streamlit example")

st.write("""
# explore new features
### new text in small 
""")

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

st.write("shape",x.shape)
st.write("classes are", len(np.unique(y)))

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

add_parameter_uci(classifier_name)

def get_classifier(clf_name, params):
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
    