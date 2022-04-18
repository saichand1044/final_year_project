import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache(persist= True)
def load():
    data= pd.read_excel('C:/Users/saich/OneDrive/Desktop/Diabetes_Classification.xlsx')
    data.drop(labels="Patient number",axis=1,inplace=True)
    data.replace({'Gender':{"female":"F","male":"M"}},inplace=True)
    le=LabelEncoder()
    data["Gender"]=le.fit_transform(data["Gender"])
    return data



def app():
    df = load()
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.sidebar.header("Want to know how the data looks???")
    if st.sidebar.button("Display Data", key="Display"):
        st.write("The following is the DataFrame of the `Diabetes` dataset.")
        st.write(df)
    if st.sidebar.button("Attributes", key="Attributes"):
        st.write(df.columns)
    if st.sidebar.button("Target", key="target"):
        st.write("Diabetes",df[df["Diabetes"]=="Diabetes"].count()["Diabetes"])
        st.write("No Diabetes",df[df["Diabetes"]=="No diabetes"].count()["Diabetes"])
    if st.sidebar.button("Profiling",False):
        profile = ProfileReport(df,title="Agriculture Data") 
        st_profile_report(profile)
    if st.sidebar.button("Discriptive Statistics",False):
        st.write(df.iloc[:,:-1].describe(include='all'))
    if st.sidebar.button("Correlation",False):
        corr = df.corr()
        plt.figure(figsize=(10,10))
        sns.heatmap(corr, annot=True, square=True)
        plt.yticks(rotation=0)
        st.pyplot()    