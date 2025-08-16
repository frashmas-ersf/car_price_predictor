import streamlit as st
import pandas as pd
import pickle
st.set_page_config(page_title='car_price_prediction')
st.header('Welcome to car price prediction app')
df=pd.read_csv('cleaned_data.csv')
Make=list(df['Make'].unique())
Make.sort()
Model=list(df['Model'].unique())
Model.sort()
Year=list(df['Year'].unique())
Year.sort()
FuelType=list(df['Fuel Type'].unique())
FuelType.sort()
Transmission=list(df['Transmission'].unique())
Transmission.sort()
with open('Adaboost_model.pkl','rb') as file:
    Adaboost_model=pickle.load(file)
with st.container(border=True):
    Make=st.selectbox('Make',options=Make)
    Model=st.selectbox('Model',options=Model)
    Year=st.selectbox('Year',options=Year)
    Engine_Size=st.number_input('Engine Size',min_value=0.0)
    Mileage=st.number_input('Mileage',min_value=0)
    FuelType=st.radio('Fuel Type',options=FuelType)
    Transmission=st.selectbox('Transmission',options=Transmission)
    input_values=[(Make.index(Make),Model.index(Model),Year,Engine_Size,Mileage,FuelType.index(FuelType),Transmission.index(Transmission))]
    if st.button('Predict price'):
        out=Adaboost_model.predict(input_values)
        st.subheader(f'Total_price: {out[0]}')

    


    
