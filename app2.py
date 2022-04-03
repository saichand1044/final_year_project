from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,make_scorer,accuracy_score
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import ADASYN
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier 
import sklearn
from sklearn.model_selection import cross_validate,cross_val_score,KFold


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
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.35, random_state=0)
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split(X,Y)

count=len(y_train)+len(y_test)


if st.sidebar.checkbox("Display size of the data", False):
    st.subheader("Size of the dataset")
    st.write(len(y_test[y_test=="Diabetes"]) )


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
        
def evaluate_model_accuracy(model, X, y):
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
    return scores
    
def evaluate_model_precision(model,X,y):
    scoring = {'precision' : make_scorer(recall_score,pos_label="Diabetes") }
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    results =cross_validate(estimator=model,
                                    X=X,
                                    y=y,
                                    cv=kfold,
                                    scoring=scoring)
    return results["test_precision"]

def evaluate_model_recall_positive(model,X,y):
    scoring = {'recall' : make_scorer(recall_score,pos_label="Diabetes") }
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    results =cross_validate(estimator=model,
                                    X=X,
                                    y=y,
                                    cv=kfold,
                                    scoring=scoring)
    return results["test_recall"]


def evaluate_model_roc_auc(model, X, y):
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=kfold, n_jobs=-1)
    return scores

class_names = ['Diabetes', 'No Diabetes']
st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest","Decision Tree",
                                                "KNN","Naive Bayes","Voting classifier","Dynamic Weighted","AdaBoostClassifier","ExtraTreesClassifier","GradientBoostingClassifier"))

if classifier == "Support Vector Machine (SVM)":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Machine (SVM) results")
        model = SVC()
        model.fit(x_train, y_train)
        accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy)
        st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train)))
        st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
        st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
        a = st.number_input('a')
        s = st.number_input('s')
        d = st.number_input('d')
        f = st.number_input('f')
        g = st.number_input('g')
        h = st.number_input('h')
        j = st.number_input('j')
        k = st.number_input('k')
        l = st.number_input('l')
        z = st.number_input('z')
        x = st.number_input('x')
        c = st.number_input('c')
        v = st.number_input('v')
        b = st.number_input('b')
        st.write("the prediction is:",list(model.predict([[a,s,d,f,g,h,j,k,l,z,x,c,v,b]]))[0])
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
        model = KNeighborsClassifier()
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
        model = VotingClassifier(estimators=[ ('dt', DecisionTreeClassifier()),('knn',KNeighborsClassifier()),('svm',SVC(probability=True)),('nb',GaussianNB()),('LR',LogisticRegression())], voting='soft')
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        plot_metrics(metrics)        
       
    
# if classifier == "Dynamic":
#     st.sidebar.subheader("Hyperparameters")
#     Dynamic_options = st.sidebar.multiselect("What dynamic classifier you want",( ("KNN","Decision Tree")))
#     if st.sidebar.button("Classify", key="classify"):
#         st.subheader("Dynamic results")
#         def dynamic_ensembles(Dynamic_options):
#             d=[]
#             if (Dynamic_options.count("KNN")==1):
#                     d.append(KNeighborsClassifier())
#             if (Dynamic_options.count("Decision Tree")==1):
#                     d.append(DecisionTreeClassifier())
#             return d
#         dy=dynamic_ensembles(Dynamic_options)
#         model = VotingClassifier(estimators=dy)
#         model.fit(x_train, y_train)
#         accuracy = model.score(x_test, y_test)
#         y_pred = model.predict(x_test)
#         st.write("Accuracy: ", accuracy.round(2))
#         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
#         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
#         plot_metrics(metrics)     

# def dynamic_ensembles(Dynamic_options):
#     d=[]
#     for i in range(len(Dynamic_options)):
#         if Dynamic_options.contains("KNN"):
#             d.append(KNeighborsClassifier())
#         if Dynamic_options.contains("Decision Tree"):
#             d.append(DecisionTreeClassifier())
#     return d
            
if classifier == "Dynamic Weighted":
    st.sidebar.subheader("Dynamic Ensemble classifiers")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    Dynamic_options = st.sidebar.multiselect("Select Your Classifiers",( ("KNN","Decision Tree","Naive bayes")))
    def dynamic_ensembles(Dynamic_options):
        d=[]
        e=[]
        if (Dynamic_options.count("KNN")==1):
            x=KNeighborsClassifier()
            d.append(tuple(("KNN",x)))
            e.append(x)
        if (Dynamic_options.count("Decision Tree")==1):
            x=DecisionTreeClassifier()
            d.append(tuple(("Decision Tree",x)))
            e.append(x)
        if (Dynamic_options.count("Naive bayes")==1):
            x=GaussianNB()
            d.append(tuple(("Naive bayes",x)))
            e.append(x)
        return d,e
    def weights(d):
        scores = list()
        for name, model in d:
            model.fit(x_train, y_train)
            yhat = model.predict(x_test)
            acc = recall_score(y_test, yhat,pos_label="Diabetes")
            scores.append(acc)
        return scores
        
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Dynamic results")
        scores = weights(dynamic_ensembles(Dynamic_options)[0])
        dy,e=dynamic_ensembles(Dynamic_options)
        model = VotingClassifier(estimators=dy,voting='soft',weights=scores)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write(scores)
        plot_metrics(metrics)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
if classifier == "AdaBoostClassifier":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("AdaBoostClassifier Results")
        model =AdaBoostClassifier()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        plot_metrics(metrics)   
        
        
        

if classifier == "GradientBoostingClassifier":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("GradientBoostingClassifier Results")
        model =GradientBoostingClassifier()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        plot_metrics(metrics)  
        
        
        
if classifier == "ExtraTreesClassifier":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("ExtraTreesClassifier Results")
        model =ExtraTreesClassifier()
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
        plot_metrics(metrics)


st.set_option('deprecation.showPyplotGlobalUse', False)        