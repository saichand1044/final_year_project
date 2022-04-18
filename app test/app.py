import streamlit as st
from multiapp import MultiApp
from apps import home, data,ML_models,project # import your app modules here

app = MultiApp()

st.markdown("""
# 🏥 DIABETES PREDICTION SYSTEM USING MACHINE LEARNING ALGORITHMS""")

# st.markdown(""" <style> .font {
# font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
# </style> """, unsafe_allow_html=True)
# st.markdown('<p class="font">🏥 DIABETES PREDICTION SYSTEM USING MACHINE LEARNING ALGORITHMS</p>', unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
# app.add_app("Random Forest", RF.app)
# app.add_app("Support Vector Machine", SVC.app)
# app.add_app("performance", performance.app)
# app.add_app("Logistic Regression", LR.app)
# app.add_app("Decision Tree", DT.app)
app.add_app("ML_models", ML_models.app)
# app.add_app("project", project.app)
# The main app
app.run()
