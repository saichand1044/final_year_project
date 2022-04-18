from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler,RobustScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,make_scorer
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import ADASYN,SMOTE
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier,StackingClassifier
import sklearn
from sklearn.model_selection import cross_validate,cross_val_score,KFold
import pandas as pd
import pandas_profiling
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from imblearn.combine import SMOTEENN,SMOTETomek

@st.cache(persist= True)
def load():
    data= pd.read_excel('C:/Users/saich/OneDrive/Desktop/Diabetes_Classification.xlsx')
    data.drop(labels="Patient number",axis=1,inplace=True)
    data.replace({'Gender':{"female":"F","male":"M"}},inplace=True)
    le=LabelEncoder()
    data["Gender"]=le.fit_transform(data["Gender"])
    return data

@st.cache(persist=True)
def outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df=df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df


@st.cache(persist=True)
def resampling(df,resample):
    target = df["Diabetes"]
    features = df.drop(columns=["Diabetes"])
    if resample=="Adasyn":
        re= ADASYN(random_state=42)
    if resample =="Smote":
        re=SMOTE(sampling_strategy='minority')
    if resample =="SMOTEENN":
        re=SMOTEENN(sampling_strategy='minority')
    if resample =="SMOTETomek":
        re=SMOTETomek(sampling_strategy='minority')    
    new_features, new_target = re.fit_resample(features, target)
    return new_features,new_target

@st.cache(persist=True)
def scaling(X,scale):
    if scale=="MinMax":
        sc= MinMaxScaler()
    if scale =="Standard":
        sc=StandardScaler()
    if scale =="Robust":
        sc=RobustScaler()    
    X_res= pd.DataFrame(sc.fit_transform(X),columns=list(X.columns))
    return X_res

@st.cache(persist=True)
def split(X,Y,size):
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=size, random_state=42)
    return x_train, x_test, y_train, y_test


def plot_metrics(metrics_list,model,x_test,y_test,class_names):
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
        

def plot_graph(plot):
    plt.figure(figsize=(10,4))
    metrics=["accuracy","recall"]
    sns.pointplot(y=plot,x=metrics,markers='*',linestyles="--")
    st.pyplot()        
        
        
def evaluate_model_accuracy(model, X, y):
    kfold = RepeatedStratifiedKFold(n_splits=10, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
    return scores
    
def evaluate_model_precision(model,X,y,validation,cross_validation):
    scoring = {'precision' : make_scorer(precision_score,pos_label="Diabetes") }
    if validation=="kfold":
        kfold = KFold(n_splits=cross_validation, shuffle=True)
        results =cross_validate(estimator=model,
                                    X=X,
                                    y=y,
                                    cv=kfold,
                                    scoring=scoring)
        return results["test_precision"]
    if validation=="stratified":
        kfold = RepeatedStratifiedKFold(n_splits=10, random_state=42)
        results =cross_validate(estimator=model,
                                    X=X,
                                    y=y,
                                    cv=kfold,
                                    scoring=scoring)
        return results["test_precision"]

def evaluate_model_recall_positive(model,X,y):
    scoring = {'recall' : make_scorer(recall_score,pos_label="Diabetes") }
    kfold = RepeatedStratifiedKFold(n_splits=10, random_state=42)
    results =cross_validate(estimator=model,
                                    X=X,
                                    y=y,
                                    cv=kfold,
                                    scoring=scoring)
    return results["test_recall"]
        

    
def evaluate_model_roc_auc(model, X, y):
    kfold = RepeatedStratifiedKFold(n_splits=10, random_state=42)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=kfold, n_jobs=-1)
    return scores


def evaluate_model_f1score(model,X,y):
    scoring = {'f1' : make_scorer(recall_score,pos_label="Diabetes") }
    kfold =RepeatedStratifiedKFold(n_splits=10, random_state=42)
    results =cross_validate(estimator=model,
                                    X=X,
                                    y=y,
                                    cv=kfold,
                                    scoring=scoring)
    return results["test_f1"]



def plot_graph(plot):
    plt.figure(figsize=(10,4))
    metrics=["accuracy","recall"]
    sns.pointplot(y=plot,x=metrics,markers='*',linestyles="--")
    st.pyplot()





def dynamic_ensembles(Dynamic_options):
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
                return d


def weights(d,x_train,y_train,x_test,y_test):
            scores = list()
            for name, model in d:
                model.fit(x_train, y_train)
                yhat = model.predict(x_test)
                acc = recall_score(y_test, yhat,pos_label="Diabetes")
                scores.append(acc)
            return scores


        
def dynamic_ensembles_weighted(Dynamic_options):
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
                
            if (Dynamic_options.count("Support Vector Machine (SVM)")==1):
                x=SVC(probability=True)
                e.append(x)
                d.append(tuple(("Support Vector Machine (SVM)",x)))
            if (Dynamic_options.count("AdaBoostClassifier")==1):
                x=AdaBoostClassifier()
                e.append(x)
                d.append(tuple(("AdaBoostClassifier",x)))
            if (Dynamic_options.count("GradientBoostingClassifier")==1):
                x=GradientBoostingClassifier()
                d.append(tuple(("GradientBoostingClassifier",x)))
                e.append(x)
            if (Dynamic_options.count("ExtraTreesClassifier")==1):
                x=ExtraTreesClassifier()
                d.append(tuple(("ExtraTreesClassifier",x)))
                e.append(x)
            return d,e
        
        

def app():
    st.title("🏥 RANDOM FOREST CLASSIFIER")
    st.sidebar.title("SIDEBAR")
    df = load()
    df=outliers(df)
    class_names = ['Diabetes', 'No Diabetes']
    st.sidebar.header('Data Preprocessing')
    size =st.sidebar.slider('Data Split(Test)', 0.15, 0.45, 0.35, 0.05)
    resample = st.sidebar.selectbox("resampling", ("Adasyn","Smote","SMOTEENN","SMOTETomek"))
    scale = st.sidebar.selectbox("scaling", ("MinMax","Standard","Robust"))
    validation=st.sidebar.selectbox("validation", ("kfold","stratified"))
    cross_validation=st.sidebar.slider('Number of Cross validation split', 2, 10)
    X, Y= resampling(df,resample) 
    X1=scaling(X,scale)
    x_train, x_test, y_train, y_test = train_test_split(X1,Y,test_size=size)
    if st.sidebar.checkbox("Type of Classifier", False):
        st.sidebar.header('Classifier')
        classifier = st.sidebar.selectbox("Which classifier you want to use", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest","Decision Tree",
                                                "KNN","Naive Bayes","Voting classifier","Dynamic","AdaBoostClassifier","ExtraTreesClassifier","GradientBoostingClassifier","Dynamic Weighted","Stacking"))
        st.sidebar.header('Plot metrics')
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)
        knn_accuracy = np.mean(evaluate_model_accuracy(knn,x_train,y_train))


        nb = GaussianNB()
        nb.fit(x_train, y_train)
        nb_accuracy = np.mean(evaluate_model_accuracy(nb,x_train,y_train))


        if classifier=="KNN":
            if st.sidebar.button("Classify", key="classify"):
                st.write('Accuracy score')
                st.info(knn_accuracy)
        if classifier=="Naive Bayes": 
            if st.sidebar.button("Classify", key="classify"):
                st.write('Accuracy score')
                st.info(nb_accuracy)
            
st.set_option('deprecation.showPyplotGlobalUse', False)        