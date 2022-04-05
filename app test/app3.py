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

# def main():
#     st.title("Diabetes Prediction using various ML algorithms")
#     st.sidebar.title("SIDEBAR")
#     st.sidebar.markdown("Want to know how the data looks????")
# if __name__ == '__main__':
#     main()

st.set_page_config(page_title="Diabetes prediction", page_icon="🏥", layout="centered")
st.title("🏥 Diabetes Prediction using various ML algorithms")
st.sidebar.title("SIDEBAR")
st.sidebar.markdown("Want to know how the data looks????")

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

    
# @st.cache(persist=True)
# def Adasyn(df):
#     target = df["Diabetes"]
#     features = df.drop(columns=["Diabetes"])
#     ada = ADASYN(random_state=42,sampling_strategy="minority")
#     new_features, new_target = ada.fit_resample(features, target)
#     return new_features,new_target


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

 

# @st.cache(persist=True)
# def scalar(X):
#     scaler = StandardScaler()
#     X_res=pd.DataFrame(scaler.fit_transform(X),columns=list(X.columns))
#     return X_res


@st.cache(persist=True)
def split(X,Y,size):
    x_train, x_test, y_train, y_test = train_test_split(X1,Y,test_size=size, random_state=42)
    return x_train, x_test, y_train, y_test

df = load()
if st.sidebar.checkbox("Display data", False):
    st.subheader("Diabetes dataset")
    st.write(df)     
if st.sidebar.checkbox("Display Attributes", False):
    st.subheader("Attributes")
    st.write(df.columns)  
if st.sidebar.checkbox("Display size of the data", False):
    st.subheader("Size of the dataset")
    st.write(len(df))    
    
profile_data=load()            
profile = ProfileReport(profile_data,title="Diabetes Data")    
if st.sidebar.checkbox("profiling",False):
    st_profile_report(profile) 

df=outliers(df)
# X, Y= Adasyn(df) 
st.sidebar.subheader("Data preprocessing")
if st.sidebar.checkbox("Data preprocessing", False):
    size = st.sidebar.number_input("Split Size", 0.1, 0.99, step=0.05, key="size")
    resample = st.sidebar.selectbox("resampling", ("Adasyn","Smote","SMOTEENN","SMOTETomek"))
    scale = st.sidebar.selectbox("scaling", ("MinMax","Standard","Robust"))
    X, Y= resampling(df,resample) 
    X1=scaling(X,scale)
    x_train, x_test, y_train, y_test = split(X1,Y,size)
    validation=st.sidebar.selectbox("validation", ("kfold","stratified"))
if st.sidebar.checkbox("Classifier", False):
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest","Decision Tree",
                                                "KNN","Naive Bayes","Voting classifier","Dynamic","AdaBoostClassifier","ExtraTreesClassifier","GradientBoostingClassifier","Dynamic Weighted","Stacking"))



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




class_names = ['Diabetes', 'No Diabetes']
st.sidebar.subheader("Choose classifier")
# classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest","Decision Tree",
#                                                 "KNN","Naive Bayes","Voting classifier","Dynamic","AdaBoostClassifier","ExtraTreesClassifier","GradientBoostingClassifier","Dynamic Weighted","Stacking"))
# validation=st.sidebar.selectbox("validation", ("kfold","stratified"))
try:
    if classifier == "Support Vector Machine (SVM)":
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine (SVM) results")
            model = SVC(probability=True)
            try:
                model.fit(x_train, y_train)
                accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy)
                st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
                st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
                st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
                st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
                plot_metrics(metrics)
            except NameError:
                st.write("Please preprocess the data")
except NameError:
    st.write("please select the classifier")
    
if classifier == "Logistic Regression":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression()
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")
        
        

if classifier == "Random Forest":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    n_estimators= st.sidebar.number_input("The number of trees in the forest", 10, 500, step=10, key="n_estimators")
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(random_state=42)
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")        
        

if classifier == "Decision Tree":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Decision Tree Results")
        model = DecisionTreeClassifier(random_state=42)
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")         
     

    
if classifier == "KNN":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    n_neighbors = st.sidebar.number_input("number of neighbors to consider are", 1, 20, step =1, key="n_neighbors")
    distance = st.sidebar.radio("distance", ("manhattan", "minkowski","euclidean"), key="distance")
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("KNN Results")
        model = KNeighborsClassifier(n_neighbors=n_neighbors,metric=distance)
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")
        
        
if classifier == "Naive Bayes":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Naive bayes Results")
        model=GaussianNB()
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")   
        
        
if classifier == "Voting classifier":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Voting Results")
        model = VotingClassifier(estimators=[ ('dt', DecisionTreeClassifier()),('knn',KNeighborsClassifier()),('svm',SVC(probability=True)),('nb',GaussianNB()),('LR',LogisticRegression())], voting='soft')
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")        
       
    

            
if classifier == "Dynamic":
    st.sidebar.subheader("Dynamic Ensemble classifiers")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    Dynamic_options = st.sidebar.multiselect("Select Your Classifiers",( ("KNN","Decision Tree","Naive bayes","Support Vector Machine (SVM)","AdaBoostClassifier","GradientBoostingClassifier","ExtraTreesClassifier")))
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
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Dynamic results")
        dy=dynamic_ensembles(Dynamic_options)
        model = VotingClassifier(estimators=dy,voting='soft')
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")
        
        
        
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
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")   
        
        
        
        
if classifier == "Stacking":
    st.sidebar.subheader("Dynamic Ensemble classifiers")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    level1=st.sidebar.radio("select your Level-1 classifier", ["KNN","Decision Tree","Naive bayes","Support Vector Machine (SVM)","AdaBoostClassifier","GradientBoostingClassifier","ExtraTreesClassifier","Random Forest","Logistic Regression"], key="level1")
    
    list_classifiers=["KNN","Decision Tree","Naive bayes","Support Vector Machine (SVM)","AdaBoostClassifier","GradientBoostingClassifier","ExtraTreesClassifier","Random Forest","Logistic Regression"]
    list_classifiers.remove(level1)
    Dynamic_options = st.sidebar.multiselect("Select Your Level-0 Classifiers", list_classifiers)
    def dynamic_ensembles(Dynamic_options,level1):
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
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Dynamic results")
        dy,a=dynamic_ensembles(Dynamic_options,level1)
        try:
            try:
                model = StackingClassifier(estimators=dy,final_estimator=a)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train)))
                st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
                st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
                st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
                plot_metrics(metrics) 
            except ValueError:
                st.error("Please select alteast one Level-0 classifier")
        except NameError:
            st.write("Please preprocess the data")
#         model.fit(x_train, y_train)
#         accuracy = model.score(x_test, y_test)
#         y_pred = model.predict(x_test)
#         st.write("Accuracy: ", accuracy.round(2))
#         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
#         st.write("Recall: ",  recall_score(y_test, y_pred, labels=class_names,pos_label="Diabetes").round(2))
#         st.write(Dynamic_options)
#         plot_metrics(metrics)        
        
        
        
        
        
if classifier == "AdaBoostClassifier":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("AdaBoostClassifier Results")
        model =AdaBoostClassifier(random_state=42,n_estimators=250,learning_rate=1)
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")   
        
        
        

if classifier == "GradientBoostingClassifier":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("GradientBoostingClassifier Results")
        model =GradientBoostingClassifier(n_estimators=250,random_state=42,learning_rate=0.9)
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")  
        
        
        
if classifier == "ExtraTreesClassifier":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("ExtraTreesClassifier Results")
        model =ExtraTreesClassifier(n_estimators=250,max_features=5,random_state=42)
        try:
            model.fit(x_train, y_train)
            accuracy = np.mean(evaluate_model_accuracy(model,x_train,y_train))
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("Precision: ", np.mean(evaluate_model_precision(model,x_train,y_train,validation)))
            st.write("Recall: ", np.mean(evaluate_model_recall_positive(model,x_train,y_train)))
            st.write("roc_auc: ", np.mean(evaluate_model_roc_auc(model,x_train,y_train)))
            st.write("F1_score: ", np.mean(evaluate_model_f1score(model,x_train,y_train)))
            plot_metrics(metrics)
        except NameError:
            st.write("Please preprocess the data")
                
            
            
st.set_option('deprecation.showPyplotGlobalUse', False)