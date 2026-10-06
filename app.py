import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor | Live AI Valuation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Banner / Hero */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        border-radius: 18px;
        padding: 32px 36px;
        margin-bottom: 28px;
        color: white;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 0px;
    }
    
    /* Glassmorphism Cards */
    .card-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(226, 232, 240, 0.15);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Valuation Price Box */
    .price-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 18px;
        padding: 30px 24px;
        text-align: center;
        box-shadow: 0 12px 35px rgba(16, 185, 129, 0.35);
        animation: pulseGlow 2s infinite alternate;
    }
    .price-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        opacity: 0.9;
        margin-bottom: 6px;
    }
    .price-value {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .price-words {
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 6px;
        opacity: 0.95;
    }
    .price-range {
        font-size: 0.88rem;
        opacity: 0.85;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Specs pills */
    .spec-pill {
        display: inline-block;
        background: rgba(148, 163, 184, 0.15);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* Primary Predict Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 14px 28px;
        border-radius: 12px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.45);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Loaders (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    search_dirs = [
        os.path.dirname(__file__),
        os.getcwd(),
        '/Users/mdinzemam/Downloads/car-project',
        '/Users/mdinzemam/.gemini/antigravity-ide/scratch/car-price-predictor'
    ]
    
    for directory in search_dirs:
        pkl_path = os.path.join(directory, 'LinearRegressionModel.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
                
        joblib_path = os.path.join(directory, 'LinearRegressionModel.joblib')
        if os.path.exists(joblib_path):
            import joblib
            return joblib.load(joblib_path)
            
    raise FileNotFoundError("Model file (LinearRegressionModel.pkl) could not be located.")

@st.cache_data
def load_data():
    search_dirs = [
        os.path.dirname(__file__),
        os.getcwd(),
        '/Users/mdinzemam/Downloads/car-project',
        '/Users/mdinzemam/.gemini/antigravity-ide/scratch/car-price-predictor'
    ]
    for directory in search_dirs:
        csv_path = os.path.join(directory, 'Cleaned_Car_data.csv')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
            
    raw_csv = '/Users/mdinzemam/Downloads/quikr_car.csv'
    if os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
        df = df[df['year'].str.isnumeric()]
        df['year'] = df['year'].astype(int)
        df = df[df['Price'] != 'Ask For Price']
        df['Price'] = df['Price'].str.replace(',', '').astype(int)
        df['kms_driven'] = df['kms_driven'].str.split().str.get(0).str.replace(',', '')
        df = df[df['kms_driven'].str.isnumeric()]
        df['kms_driven'] = df['kms_driven'].astype(int)
        df = df[~df['fuel_type'].isna()]
        df['name'] = df['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
        df = df.reset_index(drop=True)
        df = df[df['Price'] < 6000000]
        return df

    raise FileNotFoundError("Cleaned car data CSV could not be found.")

# Load resources
try:
    model = load_model()
    data = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Helper function for currency formatting in Indian format
def format_inr(number):
    try:
        val = int(round(number))
        if val >= 10000000:
            crores = val / 10000000
            words = f"₹ {crores:.2f} Crores"
        elif val >= 100000:
            lakhs = val / 100000
            words = f"₹ {lakhs:.2f} Lakhs"
        else:
            words = f"₹ {val:,}"
            
        s = str(val)
        if len(s) <= 3:
            numeric_formatted = s
        else:
            last3 = s[-3:]
            remaining = s[:-3]
            groups = []
            while len(remaining) > 2:
                groups.insert(0, remaining[-2:])
                remaining = remaining[:-2]
            if remaining:
                groups.insert(0, remaining)
            numeric_formatted = ",".join(groups) + "," + last3
            
        return f"₹ {numeric_formatted}", words
    except Exception:
        return f"₹ {number:,.2f}", ""

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🚗</span>
        <span>Car Price Predictor</span>
    </div>
    <p class="hero-subtitle">
        Accurate ML-powered resale value valuation based on Company, Model, Age, Mileage, and Fuel Type.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar - Presets & About
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚡ Quick Presets")
    st.write("Click any preset to load popular configurations:")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        swift_btn = st.button("Maruti Swift")
        i20_btn = st.button("Hyundai i20")
    with col_p2:
        city_btn = st.button("Honda City")
        innova_btn = st.button("Toyota Innova")
        
    st.markdown("---")
    st.subheader("ℹ️ Model Architecture")
    st.markdown("""
    - **Algorithm:** Linear Regression Pipeline
    - **Encoding:** OneHotEncoder (Categories aligned)
    - **Evaluation:** $R^2$ Score ~ 0.89 - 0.92
    - **Dataset:** Quikr Cars (Cleaned)
    """)
    st.markdown("---")
    st.caption("Crafted with Streamlit & Scikit-Learn")

# Session state initialization for presets
if "selected_company" not in st.session_state:
    st.session_state.selected_company = "Maruti"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Maruti Suzuki Swift"
if "selected_year" not in st.session_state:
    st.session_state.selected_year = 2019
if "selected_kms" not in st.session_state:
    st.session_state.selected_kms = 35000
if "selected_fuel" not in st.session_state:
    st.session_state.selected_fuel = "Petrol"

# Update values if presets are clicked
if swift_btn:
    st.session_state.selected_company = "Maruti"
    st.session_state.selected_model = "Maruti Suzuki Swift"
    st.session_state.selected_year = 2019
    st.session_state.selected_kms = 40000
    st.session_state.selected_fuel = "Petrol"
elif i20_btn:
    st.session_state.selected_company = "Hyundai"
    st.session_state.selected_model = "Hyundai Elite i20"
    st.session_state.selected_year = 2018
    st.session_state.selected_kms = 52000
    st.session_state.selected_fuel = "Petrol"
elif city_btn:
    st.session_state.selected_company = "Honda"
    st.session_state.selected_model = "Honda City"
    st.session_state.selected_year = 2017
    st.session_state.selected_kms = 60000
    st.session_state.selected_fuel = "Petrol"
elif innova_btn:
    st.session_state.selected_company = "Toyota"
    st.session_state.selected_model = "Toyota Innova 2.5"
    st.session_state.selected_year = 2015
    st.session_state.selected_kms = 95000
    st.session_state.selected_fuel = "Diesel"

# ---------------------------------------------------------
# Main Form Layout
# ---------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.subheader("🛠️ Car Specifications")
    
    # 1. Company
    companies = sorted(data['company'].unique())
    company_index = companies.index(st.session_state.selected_company) if st.session_state.selected_company in companies else 0
    selected_company = st.selectbox(
        "Select Car Brand / Company:",
        options=companies,
        index=company_index,
        key="company_input"
    )
    
    # 2. Model (Filtered dynamically by company)
    filtered_models = sorted(data[data['company'] == selected_company]['name'].unique())
    if not len(filtered_models):
        filtered_models = sorted(data['name'].unique())
        
    model_index = 0
    if st.session_state.selected_model in filtered_models:
        model_index = filtered_models.index(st.session_state.selected_model)
        
    selected_model = st.selectbox(
        "Select Car Model:",
        options=filtered_models,
        index=model_index,
        key="model_input"
    )
    
    # 3. Year & Fuel Type in 2 columns
    c_year, c_fuel = st.columns(2)
    with c_year:
        years = sorted(data['year'].unique(), reverse=True)
        year_index = years.index(st.session_state.selected_year) if st.session_state.selected_year in years else 0
        selected_year = st.selectbox(
            "Registration Year:",
            options=years,
            index=year_index,
            key="year_input"
        )
    with c_fuel:
        fuels = sorted(data['fuel_type'].unique())
        fuel_index = fuels.index(st.session_state.selected_fuel) if st.session_state.selected_fuel in fuels else 0
        selected_fuel = st.selectbox(
            "Fuel Type:",
            options=fuels,
            index=fuel_index,
            key="fuel_input"
        )
        
    # 4. Kilometers Driven
    selected_kms = st.number_input(
        "Kilometers Driven (Total Mileage in Kms):",
        min_value=100,
        max_value=1000000,
        value=int(st.session_state.selected_kms),
        step=5000,
        key="kms_input"
    )
    
    predict_button = st.button("🔮 Predict Estimated Resale Price")

# ---------------------------------------------------------
# Prediction Computation & Output
# ---------------------------------------------------------
with col_right:
    st.subheader("💡 Valuation Result")
    
    # Prepare input DataFrame exactly matching pipeline expectation
    input_df = pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([selected_model, selected_company, selected_year, selected_kms, selected_fuel]).reshape(1, 5)
    )
    
    try:
        raw_prediction = model.predict(input_df)[0]
        # Guarantee no negative pricing
        predicted_price = max(15000.0, float(raw_prediction))
        formatted_price, words_price = format_inr(predicted_price)
        
        # Price ranges (± 6% valuation bracket)
        lower_bound = max(10000.0, predicted_price * 0.94)
        upper_bound = predicted_price * 1.06
        fmt_low, _ = format_inr(lower_bound)
        fmt_high, _ = format_inr(upper_bound)
        
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">Estimated Market Valuation</div>
            <div class="price-value">{formatted_price}</div>
            <div class="price-words">({words_price})</div>
            <div class="price-range">
                Typical Dealer & Private Range: <strong>{fmt_low}</strong> – <strong>{fmt_high}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.markdown("#### 📋 Selected Vehicle Summary")
        st.markdown(f"""
        <div>
            <span class="spec-pill">🏷️ {selected_company}</span>
            <span class="spec-pill">🚘 {selected_model}</span>
            <span class="spec-pill">📅 Year: {selected_year}</span>
            <span class="spec-pill">🛣️ {selected_kms:,} kms</span>
            <span class="spec-pill">⛽ {selected_fuel}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Contextual insights based on inputs
        with st.expander("🔍 Factors Influencing this Valuation", expanded=True):
            age = max(0, 2026 - selected_year)
            st.write(f"- **Vehicle Age:** ~{age} years old (Depreciation rate reflects model year).")
            st.write(f"- **Usage Profile:** {selected_kms:,} total kilometers driven.")
            st.write(f"- **Fuel Efficiency & Demand:** {selected_fuel} variants reflect market liquidity.")
            
    except Exception as err:
        st.error(f"Prediction calculation error: {err}")

# ---------------------------------------------------------
# Dataset Explorer & Stats Tab
# ---------------------------------------------------------
st.markdown("---")
with st.expander("📊 Explore Market Dataset & Sample Vehicles"):
    tab1, tab2 = st.tabs(["Company Averages", "Recent Database Records"])
    with tab1:
        avg_prices = data.groupby('company')['Price'].mean().sort_values(ascending=False).head(10)
        st.write("Top 10 Car Brands by Average Selling Price in Dataset:")
        st.bar_chart(avg_prices)
    with tab2:
        st.dataframe(
            data[['company', 'name', 'year', 'kms_driven', 'fuel_type', 'Price']].head(15),
            use_container_width=True
        )
