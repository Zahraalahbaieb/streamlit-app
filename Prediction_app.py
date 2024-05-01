import streamlit as st
from joblib import load

# Make sure 'model.joblib' is the correct path to your saved model
model = load(open("gradient_boosting_model.joblib","rb"))
#Grouping features for better organization in the UI


feature_groups =['Company_ID',
       'Internet Activity Score', 'Industry', 'Employee Count',
       'Has the team size grown', 'Number of Investors in Seed',
       'Number of Co-founders', 'Number of of advisors',
       'Team size Senior leadership',
       'Presence of a top angel or venture fund in previous round of investment',
       'Number of of repeat investors', 'Number of  Sales Support material',
       'Worked in top companies',
       'Average size of companies worked for in the past',
       'Was he or she partner in Big 5 consulting?', 'Consulting experience?',
       'Catering to product/service across verticals',
       'Focus on consumer data?', 'Cloud or platform based serive/product?',
       'Local or global player', 'Crowdfunding based business',
       'Machine Learning based business', 'Predictive Analytics business',
       'Big Data Business', 'Cross-Channel Analytics/ marketing channels',
       'Is the company an aggregator/market place? e.g. Bluekai',
       'B2C or B2B venture?', 'How much is it being talked about?',
       'Average Years of experience for founder and co founder',
       'Highest education', 'Specialization of highest education',
       'Relevance of education to venture',
       'Experience in selling and building products',
       'Experience in Fortune 100 organizations',
       'Experience in Fortune 500 organizations',
       'Experience in Fortune 1000 organizations',
       'Number of Recognitions for Founders and Co-founders',
       'Team Composition score', 'Pricing Strategy',
       'Time to market service or product',
       'Long term relationship with other founders',
       'Proprietary or patent position (competitive position)',
       'Barriers of entry for the competitors', 'Company awards',
       'Legal risk and intellectual property',
       'Technical proficiencies to analyse and interpret unstructured data',
       'Solutions offered', 'Invested through global incubation competitions?',
       'Disruptiveness of technology', 'Number of Direct competitors',
       'Employees per year of company existence',
       'Avg time to investment - average across all rounds, measured from previous investment',
       'GArtsner hype cycle stage',
       'Time to maturity of Technology (in years)', 'Percent_skill_Operations',
       'Percent_skill_Data Science', 'Percent_skill_Business Strategy']

 
# Initialize session state variables for navigation and page tracking



user_input={}

for feature in feature_groups:
    f=st.number_input(f'{feature}')
    user_input[feature]=f


       
import pandas as pd
   
df_pred = pd.DataFrame(user_input,index=[0])

submitted=st.button("next")
if submitted:
    prediction=model.predict(df_pred)
    if prediction[0]==0:
        st.success('No')
    else:
        st.success('Yes')


        



