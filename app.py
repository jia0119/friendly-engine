import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Customer Churn Analytics",
    page_icon="📶",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_sample_data():
    """Generates a synthetic Telco Customer Churn dataset for immediate demo use."""
    np.random.seed(42)
    n = 1000
    
    genders = np.random.choice(["Male", "Female"], size=n)
    senior = np.random.choice([0, 1], size=n, p=[0.84, 0.16])
    tenure = np.random.randint(1, 72, size=n)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    paperless = np.random.choice(["Yes", "No"], size=n)
    payment = np.random.choice([
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], size=n)
    monthly_charges = np.random.uniform(18.25, 118.75, size=n).round(2)
    total_charges = (tenure * monthly_charges + np.random.normal(0, 10, size=n)).round(2)
    total_charges = np.maximum(total_charges, monthly_charges)
    
    # Calculate synthetic churn probabilities
    churn_prob = (
        0.4 * (contract == "Month-to-month") +
        0.3 * (payment == "Electronic check") -
        0.005 * tenure +
        0.003 * monthly_charges
    )
    churn_prob = (churn_prob - churn_prob.min()) / (churn_prob.max() - churn_prob.min())
    churn = np.where(churn_prob > 0.5, "Yes", "No")

    df = pd.DataFrame({
        "gender": genders,
        "SeniorCitizen": senior,
        "tenure": tenure,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })
    return df

@st.cache_resource
def train_model(df):
    """Preprocesses dataset and trains a Random Forest Classifier."""
    data = df.copy()
    label_encoders = {}
    
    categorical_cols = ["gender", "Contract", "PaperlessBilling", "PaymentMethod"]
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
        
    X = data.drop("Churn", axis=1)
    y = (data["Churn"] == "Yes").astype(int)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, label_encoders

df = load_sample_data()
model, encoders = train_model(df)

# -----------------------------------------------------------------------------
# 3. Main Interface Header
# -----------------------------------------------------------------------------
st.title("📶 Telecom Customer Churn Prediction & Analytics")
st.write("Analyze churn drivers, visualize customer demographics, and predict churn risk in real-time.")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Analytics Dashboard", "🔮 Predict Churn", "📁 Batch Prediction"])

# -----------------------------------------------------------------------------
# Tab 1: EDA & KPI Dashboard
# -----------------------------------------------------------------------------
with tab1:
    st.header("Executive Summary")
    
    # Top KPI Metrics
    total_customers = len(df)
    churn_count = (df["Churn"] == "Yes").sum()
    churn_rate = (churn_count / total_customers) * 100
    avg_monthly = df["MonthlyCharges"].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Total Churned", f"{churn_count:,}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Avg Monthly Charge", f"${avg_monthly:.2f}")
    
    st.markdown("---")
    
    # Interactive Plots
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Churn Distribution by Contract Type")
        fig_contract = px.histogram(
            df, x="Contract", color="Churn", barmode="group",
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"}
        )
        st.plotly_chart(fig_contract, use_container_width=True)
        
    with col_right:
        st.subheader("Monthly Charges vs Tenure")
        fig_scatter = px.scatter(
            df, x="tenure", y="MonthlyCharges", color="Churn",
            labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Charges ($)"},
            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 2: Single Customer Prediction
# -----------------------------------------------------------------------------
with tab2:
    st.header("Customer Churn Risk Assessment")
    st.write("Enter the customer details below to estimate churn probability.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        
    with col2:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=65.0)
        total_charges = st.number_input("Total Charges ($)", min_value=15.0, max_value=9000.0, value=780.0)

    if st.button("Predict Churn Risk", type="primary"):
        input_data = pd.DataFrame([{
            "gender": encoders["gender"].transform([gender])[0],
            "SeniorCitizen": senior,
            "tenure": tenure,
            "Contract": encoders["Contract"].transform([contract])[0],
            "PaperlessBilling": encoders["PaperlessBilling"].transform([paperless])[0],
            "PaymentMethod": encoders["PaymentMethod"].transform([payment])[0],
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])
        
        prob = model.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        st.subheader("Prediction Outcome")
        
        if prob > 0.5:
            st.error(f"⚠️ **High Churn Risk!** Probability: **{prob * 100:.1f}%**")
            st.warning("Recommendation: Offer a long-term contract discount or tech support incentive.")
        else:
            st.success(f"✅ **Low Churn Risk.** Probability: **{prob * 100:.1f}%**")
            st.info("Recommendation: Customer is stable. Standard retention tracking applies.")

# -----------------------------------------------------------------------------
# Tab 3: Batch CSV Prediction
# -----------------------------------------------------------------------------
with tab3:
    st.header("Batch Churn Scoring")
    uploaded_file = st.file_uploader("Upload CSV containing customer data", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("**Uploaded Data Preview:**")
        st.dataframe(batch_df.head(), use_container_width=True)
        
        if st.button("Run Batch Prediction"):
            processed_df = batch_df.copy()
            for col in ["gender", "Contract", "PaperlessBilling", "PaymentMethod"]:
                if col in processed_df.columns:
                    processed_df[col] = encoders[col].transform(processed_df[col])
            
            features = ["gender", "SeniorCitizen", "tenure", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
            probs = model.predict_proba(processed_df[features])[:, 1]
            
            batch_df["Churn_Probability"] = (probs * 100).round(1)
            batch_df["Predicted_Churn"] = np.where(probs > 0.5, "Yes", "No")
            
            st.success("Batch scoring complete!")
            st.dataframe(batch_df, use_container_width=True)
            
            csv_output = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results CSV", csv_output, "churn_predictions.csv", "text/csv")
    else:
        st.info("Upload a CSV file structured with customer feature columns to generate batch scores.")