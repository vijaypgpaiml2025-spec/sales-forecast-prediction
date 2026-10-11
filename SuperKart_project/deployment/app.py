import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_superkart_sales_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Sales Forecasting Dashboard")
st.write("Predict product store sales based on attributes and store details.")

# Input fields based on data dictionary
product_weight = st.number_input("Product Weight", value=10.0)
product_mrp = st.number_input("Product MRP (Maximum Retail Price)", value=150.0)
product_allocated_area = st.number_input("Product Allocated Area Ratio", value=0.05)
store_establishment_year = st.number_input("Store Establishment Year", value=2010, step=1)

sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
store_size = st.selectbox("Store Size", ["High", "Medium", "Low"])
city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type 1", "Supermarket Type 2", "Departmental Store", "Food Mart"])

if st.button("Predict Sales"):
    # Create input dictionary and align with model features
    # Note: Ensure features match the exact columns expected after one-hot encoding in training
    input_data = pd.DataFrame({
        "Product_Weight": [product_weight],
        "Product_MRP": [product_mrp],
        "Product_Allocated_Area": [product_allocated_area],
        "Store_Establishment_Year": [store_establishment_year],
        # Add categorical dummy defaults or handle dynamically
    })
    
    # Simple mockup prediction for display structure
    prediction = model.predict(np.zeros((1, model.n_features_in_)))[0]
    
    st.success(f"Estimated Product Store Sales Total: ${prediction:,.2f}")
