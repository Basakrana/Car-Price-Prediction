import pickle
import pandas as pd
import streamlit as st
import numpy as np

# Load Model with error handling
@st.cache_resource
def load_model():
    try:
        # Try pickle first (most compatible)
        with open('xgboost_pipeline (1).pkl', 'rb') as f:
            model = pickle.load(f)
        return model, "pickle"
    except Exception as e1:
        try:
            # Fallback to joblib if pickle fails
            import joblib
            model = joblib.load('xgboost_pipeline (1).pkl')
            return model, "joblib"
        except Exception as e2:
            st.error(f"Failed to load model with both pickle and joblib")
            st.error(f"Pickle error: {e1}")
            st.error(f"Joblib error: {e2}")
            return None, None

# Load the model
model_XGBoost, load_method = load_model()

if model_XGBoost is None:
    st.error("❌ Could not load the model. Please check the model file and library versions.")
    st.stop()
else:
    st.success(f"✅ Model loaded successfully using {load_method}")

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

# Check if image exists before displaying
try:
    st.image("how-to-sell-a-used-car-in-india.png", use_column_width=True)
except:
    st.info("Car image not found, but the app works fine without it!")

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

# Validation function
def validate_inputs():
    errors = []
    try:
        float(age)
    except:
        errors.append("Age must be a number")
    
    try:
        float(km_driven)
    except:
        errors.append("KM Driven must be a number")
    
    try:
        float(mileage)
    except:
        errors.append("Mileage must be a number")
    
    try:
        float(engine)
    except:
        errors.append("Engine CC must be a number")
    
    try:
        float(max_power)
    except:
        errors.append("Max Power must be a number")
    
    try:
        float(seats)
    except:
        errors.append("Seats must be a number")
    
    return errors

# Prepare DataFrame
def create_dataframe():
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
        return df, None
    except Exception as e:
        return None, str(e)

# Sidebar Prediction
with st.sidebar:
    st.write("# Prediction Price of Car")
    st.info("The prediction is based on 94% accuracy ✔️")
    
    if st.button("Predict Price"):
        # Validate inputs
        validation_errors = validate_inputs()
        if validation_errors:
            st.error("Please fix these input errors:")
            for error in validation_errors:
                st.error(f"❌ {error}")
        else:
            # Create dataframe
            df, df_error = create_dataframe()
            
            if df_error:
                st.error(f"Input Error ❌: {df_error}")
            else:
                try:
                    # Make prediction
                    with st.spinner("Predicting price..."):
                        prediction = model_XGBoost.predict(df)
                        
                        # Handle both log and non-log predictions
                        if prediction[0] > 50:  # If prediction seems to be in normal scale
                            price = prediction[0]
                        else:  # If prediction seems to be in log scale
                            price = np.exp(prediction[0])
                        
                        formatted_price = f"{price:,.2f}"
                        st.success(f"## The Predicted Price is ₹ {formatted_price}")
                        
                        # Show prediction confidence
                        st.info(f"Prediction made using {load_method} loader")
                        
                except Exception as e:
                    st.error(f"Prediction failed ❌: {str(e)}")
                    st.error("This might be due to model-library version incompatibility")
                    st.info("Try retraining the model with current library versions")

# Display current versions for debugging
with st.expander("Debug Information"):
    import sys
    st.write(f"Python version: {sys.version}")
    
    try:
        import sklearn
        st.write(f"Scikit-learn version: {sklearn.__version__}")
    except:
        st.write("Scikit-learn: Not available")
    
    try:
        import xgboost
        st.write(f"XGBoost version: {xgboost.__version__}")
    except:
        st.write("XGBoost: Not available")
    
    try:
        import joblib
        st.write(f"Joblib version: {joblib.__version__}")
    except:
        st.write("Joblib: Not available")
    
    st.write(f"Model loader used: {load_method}")
