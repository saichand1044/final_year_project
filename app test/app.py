from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier,StackingClassifier
import sklearn
from imblearn.over_sampling import ADASYN,SMOTE


import pandas as pd

import pandas_profiling

import streamlit as st

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


@st.cache(persist= True)
def load():
    data= pd.read_csv('C:/Users/saich/Downloads/heart.csv')
    label= LabelEncoder()
    for i in data.columns:
        data[i] = label.fit_transform(data[i])
    return data

# @st.cache(persist=True)
# def resampling(df,resample):
#     target = df["target"]
#     features = df.drop(columns=["target"])
#     if resample=="Adasyn":
#         re= ADASYN(random_state=42)
#     if resample =="Smote":
#         re=SMOTE(sampling_strategy='minority')
#     if resample =="SMOTEENN":
#         re=SMOTEENN(sampling_strategy='minority')
#     if resample =="SMOTETomek":
#         re=SMOTETomek(sampling_strategy='minority')    
#     new_features, new_target = re.fit_resample(features, target)
#     return new_features,new_target



@st.cache(persist=True)
def split(df,size):
    y = df["target"]
    x = df.drop(columns=["target"])
    x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=size, random_state=0)
    return x_train, x_test, y_train, y_test


df = load()
st.title("Introduction to building Streamlit WebApp")
st.sidebar.title("This is the sidebar")
st.sidebar.markdown("Let’s start with binary classification!!")
if st.sidebar.checkbox("Display data", False):
    st.subheader("Diabetes dataset")
    st.write(df)    
if st.sidebar.checkbox("Display Attributes", False):
    st.subheader("Attributes")
    st.write(df.columns)    
st.sidebar.subheader("Data preprocessing")
size = st.sidebar.number_input("size", 0.11, 1.00, step=0.11, key="size")
x_train, x_test, y_train, y_test=split(df,size)




def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=   class_names)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
        
class_names = ["edible", "poisnous"]
st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest","Dynamic","Stacking"))

if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel") 
    gamma = st.sidebar.radio("Gamma (Kernal coefficient", ("scale", "auto"), key="gamma")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Machine (SVM) results")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
        plot_metrics(metrics)
    
if classifier == "Logistic Regression":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)
        
        
        
        
if classifier == "Random Forest":
    st.sidebar.subheader("Hyperparameters")
    n_estimators= st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
    max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 20, step =1, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap= bootstrap, n_jobs=-1 )
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)
        
        
        
if classifier == "Stacking":
    st.sidebar.subheader("Dynamic Ensemble classifiers")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    level1=st.sidebar.radio("select your level1 classifier", ["KNN", "Decision Tree"], key="level1")
    list_classifiers=["KNN","Decision Tree","Naive bayes","Support Vector Machine (SVM)","AdaBoostClassifier","GradientBoostingClassifier","ExtraTreesClassifier"]
    list_classifiers.remove(level1)
    Dynamic_options = st.sidebar.multiselect("Select Your Classifiers", list_classifiers)
    def dynamic_ensembles(Dynamic_options,level1):
            d=[]
            if (Dynamic_options.count("KNN")==1):
                x=KNeighborsClassifier()
                d.append(tuple(("KNN",x)))
            if (Dynamic_options.count("Decision Tree")==1):
                x=DecisionTreeClassifier()
                d.append(tuple(("Decision Tree",x)))
            if (Dynamic_options.count("Naive bayes")==1):
                x=GaussianNB()
                d.append(tuple(("Naive bayes",x)))
            if (Dynamic_options.count("Support Vector Machine (SVM)")==1):
                x=SVC(probability=True)
                d.append(tuple(("Support Vector Machine (SVM)",x)))
            if (Dynamic_options.count("AdaBoostClassifier")==1):
                x=AdaBoostClassifier()
                d.append(tuple(("AdaBoostClassifier",x)))
            if (Dynamic_options.count("GradientBoostingClassifier")==1):
                x=GradientBoostingClassifier()
                d.append(tuple(("GradientBoostingClassifier",x)))
            if (Dynamic_options.count("ExtraTreesClassifier")==1):
                x=ExtraTreesClassifier()
                d.append(tuple(("ExtraTreesClassifier",x)))
            if (level1=="KNN"):
                a=KNeighborsClassifier()
            return d,a
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Dynamic results")
        dy,a=dynamic_ensembles(Dynamic_options,level1)
        model = StackingClassifier(estimators=dy,final_estimator=a)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        st.write(Dynamic_options)
        plot_metrics(metrics)
        

        
        
# if classifier == "Dynamic":
#     st.sidebar.subheader("Dynamic Ensemble classifiers")
#     metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
#     list_classifiers=["KNN","Decision Tree","Naive bayes","Support Vector Machine (SVM)","AdaBoostClassifier","GradientBoostingClassifier","ExtraTreesClassifier"]
#     Dynamic_options = st.sidebar.multiselect("Select Your Classifiers", list_classifiers)
#     def dynamic_ensembles(Dynamic_options):
#             d=[]
#             if (True):
#                 Dynamic_options.remove(level1)
#             if (Dynamic_options.count("KNN")==1):
#                 x=KNeighborsClassifier()
#                 d.append(tuple(("KNN",x)))
#             if (Dynamic_options.count("Decision Tree")==1):
#                 x=DecisionTreeClassifier()
#                 d.append(tuple(("Decision Tree",x)))
#             if (Dynamic_options.count("Naive bayes")==1):
#                 x=GaussianNB()
#                 d.append(tuple(("Naive bayes",x)))
            
            
#             if (Dynamic_options.count("Support Vector Machine (SVM)")==1):
#                 x=SVC(probability=True)
#                 d.append(tuple(("Support Vector Machine (SVM)",x)))
#             if (Dynamic_options.count("AdaBoostClassifier")==1):
#                 x=AdaBoostClassifier()
#                 d.append(tuple(("AdaBoostClassifier",x)))
#             if (Dynamic_options.count("GradientBoostingClassifier")==1):
#                 x=GradientBoostingClassifier()
#                 d.append(tuple(("GradientBoostingClassifier",x)))
#             if (Dynamic_options.count("ExtraTreesClassifier")==1):
#                 x=ExtraTreesClassifier()
#                 d.append(tuple(("ExtraTreesClassifier",x)))
#             return d
#     if st.sidebar.button("Classify", key="classify"):
#         st.subheader("Dynamic results")
#         dy=dynamic_ensembles(Dynamic_options)
#         model = VotingClassifier(estimators=dy,voting='soft')
#         model.fit(x_train, y_train)
#         accuracy = model.score(x_test, y_test)
#         y_pred = model.predict(x_test)
#         st.write("Accuracy: ", accuracy.round(2))
#         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
#         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
#         st.write(Dynamic_options)
#         plot_metrics(metrics)        
        
        
profile = ProfileReport(df,title="Agriculture Data")   
if st.sidebar.checkbox("profiling",False):
    st_profile_report(profile)
        
        
        
if st.sidebar.checkbox("asdfasd", False):
    st.subheader("Show Mushroom dataset")
    st.write(dy)
#             for i in Dynamic_options:
#                 if i =="KNN":
#                     x=KNeighborsClassifier()
#                     d.append(tuple(("KNN",x)))