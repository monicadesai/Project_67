import streamlit as st
import pandas as pd

st.title('Predicting Salaries')

#default_dd=pd.read_csv('/Users/Monica/Desktop/salary_prediction_app/simple_data.csv')
default_dd=pd.read_csv('simple_data.csv')
#default_dd=pd.read_csv('/Users/Monica/Desktop/Projects/Python_Projects_1/23)End_To_End_Projects/Project_3_ML_Scrape_To_Deployment_EndToEnd_SalaryPrediction/Project_ML_SalaryPrediction/simple_data.csv')

_job_title=list(default_dd['Job Title'].unique())
_company_name=list(default_dd['Company Name'].unique())
_location=list(default_dd['Location'].unique())
_hq=list(default_dd['Headquarters'].unique())
_size=list(default_dd['Size'].unique())
_ownership=list(default_dd['Type of ownership'].unique())
_industry=list(default_dd['Industry'].unique())
_sector=list(default_dd['Sector'].unique())
_revenue=list(default_dd['Revenue'].unique())
_competitors=list(default_dd['Competitors'].unique())
_job_state=list(default_dd['job_state'].unique())

job_title=st.selectbox('Select Job Title',options=_job_title)
company_name=st.selectbox('Select Company Nmae',options=_company_name)
location=st.selectbox('Select Location',options=_location)
hq=st.selectbox('Select Headquarters',options=_hq)
size=st.selectbox('Select Size',options=_size)
ownership=st.selectbox('Select Type of ownership',options=_ownership)
industry=st.selectbox('Select Industry',options=_industry)
sector=st.selectbox('Select Sector',options=_sector)
revenue=st.selectbox('Select Revenue',options=_revenue)
competitors=st.selectbox('Select Competitors',options=_competitors)
job_state=st.selectbox('Select job_state',options=_job_state)

rating=st.slider('Select Rating',float(default_dd['Rating'].min()),float(default_dd['Rating'].max()))
founded=st.slider('Select Founded',float(default_dd['Founded'].min()),float(default_dd['Founded'].max()))
hourly=st.selectbox('Select hourly',options=[0,1])
employer_provided=st.selectbox('Select employer_provided',options=[0,1])
same_state=st.selectbox('Select same_state',options=[0,1])
age=founded=st.slider('Select age',float(default_dd['age'].min()),float(default_dd['age'].max()))
python_yn=st.selectbox('Select python_yn',options=[0,1])
R_yn=st.selectbox('Select R_yn',options=[0,1])
spark=st.selectbox('Select spark',options=[0,1])
aws=st.selectbox('Select aws',options=[0,1])
excel=st.selectbox('Select excel',options=[0,1])
num_comp=st.selectbox('Select num_comp',options=list(default_dd['num_comp'].unique()))

# A lot of these inputs can be formatted to look better 
# for example founding year can be made to take integer value and we can change the ranges
# for now all is coming directly from the raw data


x=pd.DataFrame({'Job Title':[job_title], 
	'Rating':[rating], 
	'Company Name':[company_name], 
	'Location':[location], 
	'Headquarters':[hq],
    'Size':[size], 
    'Founded':[founded], 
    'Type of ownership':[ownership], 
    'Industry':[industry], 
    'Sector':[sector], 
    'Revenue':[revenue],
    'Competitors':[competitors],
    'hourly':[hourly], 
    'employer_provided':[employer_provided], 
    'job_state':[job_state], 
    'same_state':[same_state],
    'age':[age], 
    'python_yn':[python_yn], 
    'R_yn':[R_yn], 
    'spark':[spark], 
    'aws':[aws], 
    'excel':[excel], 
    'num_comp':[num_comp]})

from sklearn.externals import joblib

model=open('model_pipeline.pkl','rb')
model=joblib.load(model)

st.title('Your Predicted Salary is :')
st.info('$'+str(int(model.predict(x)[0]))+'K')
