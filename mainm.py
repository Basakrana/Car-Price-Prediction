import joblib
import pandas as pd
import streamlit as st
import numpy as np

# Load Model
model_XGBoost = joblib.load('xgboost_pipelinen.pkl')

# App UI
st.title('Car Price Prediction 🚗💵')
st.markdown(
    """
    <div style="background-color:#000000; padding:10px; border-radius:5px">
        <h4 style="color:#faf7f7;">This app predicts the price of a car based on its features 🚀. 
        Select categorical features and enter numeric features manually.</h4>
    </div>
    """,
    unsafe_allow_html=True
)
st.image("how-to-sell-a-used-car-in-india.png", use_column_width=True)

# Dropdown options
list_Cars = ['Maruti','Skoda','Honda','Hyundai','Toyota','Ford','Renault','Mahindra','Tata','Chevrolet',
             'Datsun','Jeep','Mercedes-Benz','Mitsubishi','Audi','Volkswagen','BMW','Nissan','Lexus',
             'Jaguar','Land','MG','Volvo','Daewoo','Kia','Fiat','Force','Ambassador','Ashok','Isuzu','Opel']

fuel_types = ['Diesel', 'Petrol', 'LPG', 'CNG']
owner_types = ['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car']
Seller_Types = ['Individual','Dealer','Trustmark Dealer']
transmission_types = ['Manual','Automatic']

# Input Form
with st.container():
    st.header('Enter Car Details')
    col1, col2 = st.columns(2)

    with col1:
        name = st.selectbox("Brand of Car", list_Cars)
        owner = st.selectbox("Type of Owner", owner_types)
        transmission = st.selectbox("Type of Transmission", transmission_types)
        fuel = st.selectbox("Type of Fuel ⛽", fuel_types)
        seller_type = st.selectbox("Type of Seller", Seller_Types)

    with col2:
        km_driven = st.text_input("KM Driven of Car", "10000")
        mileage = st.text_input("Mileage of Car", "20")
        age = st.text_input("Age of Car", "3")
        engine = st.text_input("Engine of Car (CC)", "1200")
        max_power = st.text_input("Max Power of Car (hp)", "120")

    seats = st.text_input("Number of Seats 💺", "5")

# Prepare DataFrame
try:
    df = pd.DataFrame({
        'name':[name],
        'age':[float(age)],
        'km_driven':[float(km_driven)],
        'fuel':[fuel],
        'seller_type':[seller_type],
        'transmission':[transmission],
        'owner':[owner],
        'mileage':[float(mileage)],
        'engine':[float(engine)],
        'max_power':[float(max_power)],
        'seats':[float(seats)]
    })
except Exception as e:
    st.error(f"Input Error ❌: {e}")
    df = None

# Sidebar Prediction
with st.sidebar:
    st.write("# Prediction Price of Car")
    st.info("The Calculation is based on 94% accuracy ✔️")
    if st.button("Calculated Price"):
        if df is not None:
            try:
                price = np.exp(model_XGBoost.predict(df))[0]
                formatted_price = f"{price:,.2f}"
                st.success(f"## The Calculated Price is ₹ {formatted_price}")
            except Exception as e:
                st.error(f"Calculation failed ❌: {e}")

