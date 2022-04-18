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
        
        
def evaluate_model_accuracy(model, X, y,validation,cross_validation):
    if validation=="Stratified":
        kfold = RepeatedStratifiedKFold(n_splits=cross_validation, random_state=42)
        results = cross_val_score(model, X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
    if validation=="kfold":
        kfold = KFold(n_splits=cross_validation, shuffle=True)
        results = cross_val_score(model, X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
    return results
 
    
    
    
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
    if validation=="Stratified":
        kfold = RepeatedStratifiedKFold(n_splits=cross_validation, random_state=42)
        results =cross_validate(estimator=model,
                                    X=X,
                                    y=y,
                                    cv=kfold,
                                    scoring=scoring)
        return results["test_precision"]

def evaluate_model_recall_positive(model,X,y,validation,cross_validation):
    scoring = {'recall' : make_scorer(recall_score,pos_label="Diabetes") }
    if validation=="Stratified":
        kfold = RepeatedStratifiedKFold(n_splits=cross_validation, random_state=42)
        results =cross_validate(estimator=model,
                                    X=X,
                                    y=y,
                                    cv=kfold,
                                    scoring=scoring)
    if validation=="kfold":
        kfold = KFold(n_splits=cross_validation, shuffle=True)
        results =cross_validate(estimator=model,
                                    X=X,
                                    y=y,
                                    cv=kfold,
                                    scoring=scoring)    
    return results["test_recall"]
        

    
def evaluate_model_roc_auc(model, X, y,validation,cross_validation):
    if validation=="Stratified":
        kfold = RepeatedStratifiedKFold(n_splits=cross_validation, random_state=42)
        results = cross_val_score(model, X, y, scoring='roc_auc', cv=kfold, n_jobs=-1)
    if validation=="kfold":
        kfold = KFold(n_splits=cross_validation, shuffle=True)
        results = cross_val_score(model, X, y, scoring='roc_auc', cv=kfold, n_jobs=-1)
    return results


def evaluate_model_f1score(model,X,y,validation,cross_validation):
    scoring = {'f1' : make_scorer(recall_score,pos_label="Diabetes") }
    if validation=="Stratified":
        kfold = RepeatedStratifiedKFold(n_splits=cross_validation, random_state=42)
        results =cross_validate(estimator=model,
                                    X=X,
                                    y=y,
                                    cv=kfold,
                                    scoring=scoring)
    if validation=="kfold":
        kfold = KFold(n_splits=cross_validation, shuffle=True)
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
        

        
def dynamic_ensembles_stacking(Dynamic_options,level1):
                d=[]
                if (Dynamic_options.count("KNN")==1):
                    x=KNeighborsClassifier()
                    d.append(tuple(("KNN",x)))
                if (Dynamic_options.count("Logistic Regression")==1):
                    x=LogisticRegression()
                    d.append(tuple(("Logistic Regression",x)))
                if (Dynamic_options.count("Decision Tree")==1):
                    x=DecisionTreeClassifier()
                    d.append(tuple(("Decision Tree",x)))
                if (Dynamic_options.count("Naive bayes")==1):
                    x=GaussianNB()
                    d.append(tuple(("Naive bayes",x)))
                if (Dynamic_options.count("Support Vector Machine (SVM)")==1):
                    x=SVC(probability=True)
                    d.append(tuple(("Support Vector Machine (SVM)",x)))
                if (Dynamic_options.count("Random Forest")==1):
                    x=RandomForestClassifier()
                    d.append(tuple(("Random Forest",x)))
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
                if (level1=="Logistic Regression"):
                    a=LogisticRegression()
                if (level1=="Decision Tree"):
                    a=DecisionTreeClassifier()
                if (level1=="Naive bayes"):
                    a=GaussianNB()
                if (level1=="Support Vector Machine (SVM)"):
                    a=SVC(probability=True)
                if (level1=="Random Forest"):
                    a=RandomForestClassifier()
                if (level1=="AdaBoostClassifier"):
                    a=AdaBoostClassifier()
                if (level1=="GradientBoostingClassifier"):
                    a=GradientBoostingClassifier()
                if (level1=="ExtraTreesClassifier"):
                    a=ExtraTreesClassifier()
                return d,a        
        
        
        

def app():
    st.sidebar.title("SIDEBAR")
    df = load()
    df=outliers(df)
    class_names = ['Diabetes', 'No Diabetes']
    st.sidebar.header('Data Preprocessing')
    size =st.sidebar.slider('Data Split(Test)', 0.15, 0.45, 0.35, 0.05)
    resample = st.sidebar.selectbox("resampling", ("Adasyn","Smote","SMOTEENN","SMOTETomek"))
    scale = st.sidebar.selectbox("scaling", ("Standard","Robust","MinMax"))
    validation=st.sidebar.selectbox("validation", ("Stratified","kfold"))
    cross_validation=st.sidebar.slider('Number of Cross validation split', 2, 15,8,1)
    X, Y= resampling(df,resample) 
    X1=scaling(X,scale)
    x_train, x_test, y_train, y_test = train_test_split(X1,Y,test_size=size)
    if st.sidebar.checkbox("Type of Classifier", False):
        st.sidebar.header('Classifier')
        classifier = st.sidebar.selectbox("Which classifier you want to use", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest","Decision Tree",
                                                "KNN","Naive Bayes","Voting classifier","Dynamic","AdaBoostClassifier","ExtraTreesClassifier","GradientBoostingClassifier","Dynamic Weighted","Stacking"))
        st.sidebar.header('Plot metrics')
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if classifier == "KNN":
            n=st.sidebar.slider('Number of Neighbours', 1, 25)
            if st.sidebar.button("Classify", key="classify1"):
                model = KNeighborsClassifier(n_neighbors=n)
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)
                
        if classifier == "Naive Bayes":
            if st.sidebar.button("Classify", key="classify"):
                model = GaussianNB()
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)
                
                
        if classifier == "Support Vector Machine (SVM)":
            st.sidebar.header('Adjust HyperParameters')
            kernel = st.sidebar.radio("Select the kernel", ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
            C=st.sidebar.number_input("Regularization parameter", 0.0,1.0, step=0.01, key="C")
            if st.sidebar.button("Classify", key="classify"):
                model = SVC(probability=True,kernel=kernel,C=C)
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)
        
        
        if classifier == "Logistic Regression":
            if st.sidebar.button("Classify", key="classify"):
                model = LogisticRegression(random_state=42)
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)
                

        if classifier == "Random Forest":
            st.sidebar.header('Adjust HyperParameters')
            parameter_criterion = st.sidebar.selectbox('criterion',('gini', 'entropy'))
            if st.sidebar.button("Classify", key="classify"):
                model = RandomForestClassifier(random_state=42,criterion=parameter_criterion)
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)
                

        if classifier == "Decision Tree":
            st.sidebar.header('Adjust HyperParameters')
            parameter_criterion = st.sidebar.selectbox('criterion',('gini', 'entropy'))
            max_depth=st.sidebar.slider('Max depth', 2, 10,6,1)
            max_features = st.sidebar.selectbox("Max features",("auto","sqrt","log2"))
            if st.sidebar.button("Classify", key="classify"):
                st.title("PERFORMANCE")
                model = DecisionTreeClassifier(max_depth=max_depth,max_features=max_features,random_state=42,criterion=parameter_criterion)
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)
                recall=np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation))
                plot=[accuracy,recall]
        
        if classifier == "AdaBoostClassifier":
            if st.sidebar.button("Classify", key="classify"):
                model = AdaBoostClassifier(random_state=42,n_estimators=250,learning_rate=1)
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)
                
        if classifier == "GradientBoostingClassifier":
            if st.sidebar.button("Classify", key="classify"):
                model = GradientBoostingClassifier(n_estimators=250,random_state=42,learning_rate=0.9)
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)
                
                
        if classifier == "ExtraTreesClassifier":
            if st.sidebar.button("Classify", key="classify"):
                model = ExtraTreesClassifier(n_estimators=250,max_features=5,random_state=42)
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)
        
        if classifier == "Dynamic":
            Dynamic_options = st.sidebar.multiselect("Select Your Classifiers",( ("KNN","Decision Tree","Naive bayes","Support Vector Machine (SVM)","AdaBoostClassifier","GradientBoostingClassifier","ExtraTreesClassifier")))
            if st.sidebar.button("Classify", key="classify"):
                st.subheader("Dynamic results")
                dy=dynamic_ensembles(Dynamic_options)
                model = VotingClassifier(estimators=dy,voting='soft')
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)
            
        if classifier == "Voting classifier":
            model = VotingClassifier(estimators=[ ('dt', DecisionTreeClassifier()),('knn',KNeighborsClassifier()),('svm',SVC(probability=True)),('nb',GaussianNB()),('LR',LogisticRegression())], voting='soft')
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            st.write("Accuracy: ", accuracy)
            
        if classifier == "Dynamic Weighted":
            Dynamic_options = st.sidebar.multiselect("Select Your Classifiers",( ("KNN","Decision Tree","Naive bayes","Support Vector Machine (SVM)","AdaBoostClassifier","GradientBoostingClassifier","ExtraTreesClassifier")))
            if st.sidebar.button("Classify", key="classify"):
                scores = weights(dynamic_ensembles_weighted(Dynamic_options)[0],x_train,y_train,x_test,y_test)
                dy,e=dynamic_ensembles_weighted(Dynamic_options)
                model = VotingClassifier(estimators=dy,voting='soft',weights=scores)
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                st.write('Accuracy score')
                st.info(accuracy)
                st.write('Precision score')
                st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                st.write('Recall score')
                st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                st.write('ROC_AUC score')
                st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                st.write('F1 score')
                st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                plot_metrics(metrics,model,x_test,y_test,class_names)

        if classifier == "Stacking":
            level1=st.sidebar.radio("select your Level-1 classifier", ["KNN","Decision Tree","Naive bayes","Support Vector Machine (SVM)","AdaBoostClassifier","GradientBoostingClassifier","ExtraTreesClassifier","Random Forest","Logistic Regression"], key="level1")
            list_classifiers=["KNN","Decision Tree","Naive bayes","Support Vector Machine (SVM)","AdaBoostClassifier","GradientBoostingClassifier","ExtraTreesClassifier","Random Forest","Logistic Regression"]
            list_classifiers.remove(level1)
            Dynamic_options = st.sidebar.multiselect("Select Your Level-0 Classifiers", list_classifiers)
            if st.sidebar.button("Classify", key="classify"):
                dy,a=dynamic_ensembles_stacking(Dynamic_options,level1)
                try:
                    model = StackingClassifier(estimators=dy,final_estimator=a)
                    model.fit(x_train, y_train)
                    accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train,validation,cross_validation))
                    st.write('Accuracy score')
                    st.info(accuracy)
                    st.write('Precision score')
                    st.info(np.mean(evaluate_model_precision(model,x_train,y_train,validation,cross_validation)))
                    st.write('Recall score')
                    st.info(np.mean(evaluate_model_recall_positive(model,x_train,y_train,validation,cross_validation)))
                    st.write('ROC_AUC score')
                    st.info(np.mean(evaluate_model_roc_auc(model,x_train,y_train,validation,cross_validation)))
                    st.write('F1 score')
                    st.info(np.mean(evaluate_model_f1score(model,x_train,y_train,validation,cross_validation)))
                    plot_metrics(metrics,model,x_test,y_test,class_names)
                except ValueError:
                    st.error("Please select alteast one Level-0 classifier")
                    
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)       

st.set_option('deprecation.showPyplotGlobalUse', False)        