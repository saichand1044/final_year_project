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
from imblearn.over_sampling import ADASYN
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier


def main():
    st.title("Diabetes Prediction using various ML algorithms")
    st.sidebar.title("SIDEBAR")
    st.sidebar.markdown("Want to know how the data looks????")
if __name__ == '__main__':
    main()
    
    
@st.cache(persist= True)
def load():
    data= pd.read_excel('C:/Users/saich/OneDrive/Desktop/Diabetes_Classification.xlsx')
    le=LabelEncoder()
    data["Gender"]=le.fit_transform(data["Gender"])
    return data
df = load()

if st.sidebar.checkbox("Display data", False):
    st.subheader("Diabetes dataset")
    st.write(df)
    
    
@st.cache(persist=True)
def outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df=df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df
df=outliers(df)
    
@st.cache(persist=True)
def Adasyn(df):
    target = df["Diabetes"]
    features = df.drop(columns=["Diabetes"])
    ada = ADASYN(random_state=42,sampling_strategy="minority")
    new_features, new_target = ada.fit_resample(features, target)
    return new_features,new_target
X, Y= Adasyn(df)     
    
@st.cache(persist=True)
def split(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split(X,Y)

count=len(y_train)+len(y_test)


if st.sidebar.checkbox("Display size of the data", False):
    st.subheader("Size of the dataset")
    st.write(count)  


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

class_names = ['Diabetes', 'No Diabetes']
st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest","Decision Tree",
                                                "KNN","Naive Bayes","Voting classifier"))

if classifier == "Support Vector Machine (SVM)":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Machine (SVM) results")
        model = SVC()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2)) 
        plot_metrics(metrics)

if classifier == "Logistic Regression":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        plot_metrics(metrics)
        
        

if classifier == "Random Forest":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = AdaBoostClassifier()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        plot_metrics(metrics)        
        

if classifier == "Decision Tree":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Decision Tree Results")
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        plot_metrics(metrics)         
     

    
if classifier == "KNN":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Naive bayes Results")
        model = KNeighborsClassifier()()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        plot_metrics(metrics)
        
        
if classifier == "Naive Bayes":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Naive bayes Results")
        model = GaussianNB()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        plot_metrics(metrics)   
        
        
if classifier == "Voting classifier":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Voting Results")
        model = VotingClassifier(estimators=[ ('dt', DecisionTreeClassifier()),('knn',KNeighborsClassifier()),('svm', SVC(probability=True)),('nb',GaussianNB()),('LR',LogisticRegression())], voting='soft')
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        plot_metrics(metrics)        
        
  

