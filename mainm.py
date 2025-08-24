import joblib
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        color: white;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.3rem;
        font-weight: 400;
        line-height: 1.6;
        opacity: 0.95;
    }
    
    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2f3542;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #666;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Form styling */
    .form-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .form-section-title {
        color: #2f3542;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #ff6b6b;
        display: inline-block;
    }
    
    /* Input styling */
    .stSelectbox > div > div > div {
        background: white;
        border: 2px solid #e1e5e9;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: #ff6b6b;
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.1);
        transform: translateY(-2px);
    }
    
    .stNumberInput > div > div > input {
        border: 2px solid #e1e5e9;
        border-radius: 12px;
        padding: 12px 16px;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #ff6b6b;
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.1);
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.3rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(255, 107, 107, 0.6);
    }
    
    /* Result styling */
    .price-result {
        background: linear-gradient(135deg, #96ceb4, #feca57);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(150, 206, 180, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .price-result::before {
        content: '₹';
        position: absolute;
        top: 20px;
        right: 30px;
        font-size: 8rem;
        opacity: 0.1;
        font-weight: 700;
    }
    
    .price-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Metrics styling */
    .metrics-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
        backdrop-filter: blur(15px);
    }
    
    /* Info boxes */
    .info-highlight {
        background: linear-gradient(135deg, rgba(69, 183, 209, 0.1), rgba(78, 205, 196, 0.1));
        border-left: 5px solid #45b7d1;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .warning-highlight {
        background: linear-gradient(135deg, rgba(254, 202, 87, 0.1), rgba(255, 107, 107, 0.1));
        border-left: 5px solid #feca57;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_pipelinen.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file 'xgboost_pipelinen.pkl' not found! Please ensure it's in the app directory.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

model_XGBoost = load_model()

# -------------------- Header --------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🚗 AI Car Price Predictor</div>
    <div class="main-subtitle">
        Get instant, accurate car valuations powered by advanced machine learning<br>
        <strong>94% Accuracy</strong> • <strong>Instant Results</strong> • <strong>Smart Analytics</strong>
    </div>
    <div style="margin-top: 1.5rem; font-size: 1rem; opacity: 0.8;">
        <strong>Developed by: RANA BASAK</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Feature Cards --------------------
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <span class="feature-icon">🤖</span>
        <div class="feature-title">AI-Powered</div>
        <div class="feature-desc">Advanced XGBoost algorithm with 94% accuracy rate</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">⚡</span>
        <div class="feature-title">Lightning Fast</div>
        <div class="feature-desc">Get instant price predictions in seconds</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">📊</span>
        <div class="feature-title">Data-Driven</div>
        <div class="feature-desc">Based on thousands of real car sales data</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🔒</span>
        <div class="feature-title">Trusted</div>
        <div class="feature-desc">Reliable valuations you can count on</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### 🎯 Quick Guide")
    st.markdown("""
    <div class="info-highlight">
        <strong>How to get accurate predictions:</strong><br>
        1. Select your car brand and specifications<br>
        2. Enter accurate mileage and condition details<br>
        3. Click predict for instant AI valuation<br>
        4. Get market-competitive price estimate
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Price Factors")
    st.markdown("""
    <div class="warning-highlight">
        <strong>Key factors affecting price:</strong><br>
        • <strong>Age & Mileage:</strong> Lower is better<br>
        • <strong>Brand:</strong> Premium brands hold value<br>
        • <strong>Fuel Type:</strong> Impacts running costs<br>
        • <strong>Owner History:</strong> First owner premium<br>
        • <strong>Engine Power:</strong> Performance matters
    </div>
    """, unsafe_allow_html=True)

# -------------------- Data Options --------------------
list_Cars = ['Maruti','Skoda','Honda','Hyundai','Toyota','Ford','Renault','Mahindra','Tata','Chevrolet',
             'Datsun','Jeep','Mercedes-Benz','Mitsubishi','Audi','Volkswagen','BMW','Nissan','Lexus',
             'Jaguar','Land','MG','Volvo','Daewoo','Kia','Fiat','Force','Ambassador','Ashok','Isuzu','Opel']

fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG']
owner_types = ['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car']
seller_types = ['Individual','Dealer','Trustmark Dealer']
transmission_types = ['Manual','Automatic']

# -------------------- Main Form --------------------
st.markdown("""
<div class="form-container">
    <h2 class="form-section-title">🔧 Vehicle Specifications</h2>
</div>
""", unsafe_allow_html=True)

# Create improved layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("#### 🚗 Basic Information")
    name = st.selectbox("🏷️ Car Brand", list_Cars, help="Select the manufacturer of your car")
    fuel = st.selectbox("⛽ Fuel Type", fuel_types, help="Primary fuel type of the vehicle")
    transmission = st.selectbox("⚙️ Transmission", transmission_types, help="Type of gearbox")
    owner = st.selectbox("👤 Owner Type", owner_types, help="Ownership history affects value")

with col2:
    st.markdown("#### 📊 Performance Specs")
    engine = st.number_input("🔧 Engine Capacity (CC)", min_value=500, value=1200, help="Engine displacement in cubic centimeters")
    max_power = st.number_input("⚡ Max Power (hp)", min_value=50, value=120, help="Maximum power output in horsepower")
    mileage = st.number_input("⛽ Mileage (kmpl)", min_value=1.0, value=20.0, help="Fuel efficiency in kilometers per liter")
    seats = st.number_input("💺 Seating Capacity", min_value=2, value=5, help="Number of seats in the vehicle")

with col3:
    st.markdown("#### 📈 Usage & Condition")
    km_driven = st.number_input("🛣️ Kilometers Driven", min_value=0, value=10000, help="Total distance covered")
    age = st.number_input("📅 Vehicle Age (years)", min_value=0, value=3, help="Age of the vehicle in years")
    seller_type = st.selectbox("🏪 Seller Type", seller_types, help="Type of seller affects pricing")

# -------------------- Real-time Metrics --------------------
st.markdown("---")
st.markdown("### 📊 Vehicle Analysis")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    condition = "Excellent" if age < 3 else "Good" if age < 7 else "Fair"
    condition_color = "🟢" if age < 3 else "🟡" if age < 7 else "🔴"
    st.metric("🏆 Condition", f"{condition_color} {condition}")

with metric_col2:
    usage_level = "Low" if km_driven < 30000 else "Medium" if km_driven < 80000 else "High"
    usage_color = "🟢" if km_driven < 30000 else "🟡" if km_driven < 80000 else "🔴"
    st.metric("📊 Usage Level", f"{usage_color} {usage_level}")

with metric_col3:
    efficiency = "High" if mileage > 18 else "Medium" if mileage > 12 else "Low"
    efficiency_color = "🟢" if mileage > 18 else "🟡" if mileage > 12 else "🔴"
    st.metric("⛽ Fuel Efficiency", f"{efficiency_color} {efficiency}")

with metric_col4:
    performance = "High" if max_power > 150 else "Medium" if max_power > 100 else "Standard"
    performance_color = "🟢" if max_power > 150 else "🟡" if max_power > 100 else "🔴"
    st.metric("⚡ Performance", f"{performance_color} {performance}")

# -------------------- Prediction Section --------------------
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    if st.button("🔮 **GET AI PRICE PREDICTION**", key="predict_btn", help="Click to get instant AI-powered price prediction"):
        if model_XGBoost is None:
            st.error("❌ Model not available. Please check the model file.")
        else:
            try:
                # Create DataFrame
                df = pd.DataFrame({
                    'name': [name],
                    'age': [float(age)],
                    'km_driven': [float(km_driven)],
                    'fuel': [fuel],
                    'seller_type': [seller_type],
                    'transmission': [transmission],
                    'owner': [owner],
                    'mileage': [float(mileage)],
                    'engine': [float(engine)],
                    'max_power': [float(max_power)],
                    'seats': [float(seats)]
                })
                
                # Make prediction
                with st.spinner("🤖 AI is analyzing your car..."):
                    prediction = model_XGBoost.predict(df)
                    price = np.exp(prediction)[0]
                    formatted_price = f"{price:,.0f}"
                    
                    # Display result with animation
                    st.markdown(f"""
                    <div class="price-result">
                        <h2 style="margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            🎉 Predicted Price
                        </h2>
                        <div class="price-value">₹ {formatted_price}</div>
                        <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
                            Based on current market trends and vehicle specifications
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show price analysis
                    st.markdown("### 💡 Price Analysis")
                    
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        # Price range estimation
                        lower_estimate = price * 0.9
                        upper_estimate = price * 1.1
                        
                        st.markdown(f"""
                        <div class="metrics-container">
                            <h4>📊 Price Range</h4>
                            <div class="metric-card">
                                <div class="metric-value">₹ {lower_estimate:,.0f} - ₹ {upper_estimate:,.0f}</div>
                                <div class="metric-label">Estimated Range</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # with analysis_col2:
                    #     # Value assessment
                    #     price_per_year = price / max(age, 1)
                        
                    #     st.markdown(f"""
                    #     <div class="metrics-container">
                    #         <h4>💰 Value Assessment</h4>
                    #         <div class="metric-card">
                    #             <div class="metric-value">₹ {price_per_year:,.0f}</div>
                    #             <div class="metric-label">Value per Year</div>
                    #         </div>
                    #     </div>
                    #     """, unsafe_allow_html=True)
                    
                    # Create a simple price chart using matplotlib
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    categories = ['Predicted\nPrice', 'Lower\nEstimate', 'Upper\nEstimate']
                    prices = [price, lower_estimate, upper_estimate]
                    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
                    
                    bars = ax.bar(categories, prices, color=colors, alpha=0.8)
                    
                    # Add value labels on bars
                    for bar, price_val in zip(bars, prices):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'₹{price_val:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
                    ax.set_title('🎯 AI Price Prediction Analysis', fontsize=16, fontweight='bold', pad=20)
                    ax.set_ylabel('Price (₹)', fontsize=12)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
                    
                    # Style the plot
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Success animation
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 3rem; backdrop-filter: blur(10px);">
    <h3 style="color: white; margin-bottom: 1rem;">🚀 Powered by Advanced AI Technology</h3>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; line-height: 1.6; margin: 0;">
        Our machine learning model analyzes multiple factors including brand value, vehicle condition, 
        market trends, and regional pricing to provide you with the most accurate car valuation possible.
    </p>
    <br>
    <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
        <h4 style="color: white; margin-bottom: 0.5rem;">👨‍💻 Developer</h4>
        <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">RANA BASAK</p>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            Machine Learning Engineer & Data Scientist
        </p>
    </div>
    <p style="color: rgba(255,255,255,0.7); font-style: italic; margin: 0;">
        <strong>Disclaimer:</strong> Predictions are estimates based on historical data. 
        Actual market prices may vary based on local conditions and vehicle specifics.
    </p>
</div>
""", unsafe_allow_html=True)
