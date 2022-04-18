from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
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
        
        
def evaluate_model_accuracy(model, X, y):
    kfold = RepeatedStratifiedKFold(n_splits=10, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
    return scores
    
def evaluate_model_precision(model,X,y,validation):
    scoring = {'precision' : make_scorer(precision_score,pos_label="Diabetes") }
    if validation=="kfold":
        kfold = KFold(n_splits=10, shuffle=True)
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

        
def app():
    st.title("🏥 RANDOM FOREST CLASSIFIER")
    st.sidebar.title("SIDEBAR")
    df = load()
    df=outliers(df)
    class_names = ['Diabetes', 'No Diabetes']
    size = st.sidebar.number_input("Split Size", 0.1, 0.99, step=0.05, key="size")
    resample = st.sidebar.selectbox("resampling", ("Adasyn","Smote","SMOTEENN","SMOTETomek"))
    scale = st.sidebar.selectbox("scaling", ("MinMax","Standard","Robust"))
    validation=st.sidebar.selectbox("validation", ("kfold","stratified"))
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    X, Y= resampling(df,resample) 
    X1=scaling(X,scale)
    x_train, x_test, y_train, y_test = split(X1,Y,size)
    if st.sidebar.button("Classify", key="classify"):
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy)
        st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
        st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
        st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
        st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
        plot_metrics(metrics,model,x_test,y_test,class_names)
        
        
        
RF=RandomForestClassifier()        
        
        
st.set_option('deprecation.showPyplotGlobalUse', False)        