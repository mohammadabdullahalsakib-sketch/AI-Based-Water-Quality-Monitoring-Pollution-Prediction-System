import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
import streamlit.components.v1 as components

import shap

# Page Config
st.set_page_config(page_title="Water Quality AI Monitoring", layout="wide")

st.title(
    "🌊 AI-Based Water Quality Monitoring & Pollution Prediction System"
)
st.markdown(
    "Monitoring parameters: **pH, Temperature, Turbidity, Conductivity,"
    " Dissolved Oxygen (DO)**"
)


# Function to render SHAP plots in Streamlit properly
def st_shap(plot, height=None):
  shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
  components.html(shap_html, height=height)


# --- Sidebar CSV Upload Section ---
st.sidebar.header("📁 Dataset Selection")
uploaded_file = st.sidebar.file_uploader(
    "Upload Water Quality CSV", type=["csv"]
)


# --- Step 1 & 2: Load & Preprocess Data ---
@st.cache_data
def load_data(file):
  if file is not None:
    df = pd.read_csv(file)

    # Clean duplicates in original columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Mapping Kaggle columns cleanly
    column_mapping = {
        "ph": "pH",
        "Sulfate": "Conductivity",
        "Organic_carbon": "DO",
        "Potability": "Condition",
    }

    df = df.rename(columns=column_mapping)
    df = df.loc[:, ~df.columns.duplicated()]

    # Fill required missing columns
    if "Temperature" not in df.columns:
      df["Temperature"] = np.random.uniform(20.0, 35.0, len(df))
    if "DO" not in df.columns:
      df["DO"] = np.random.uniform(2.0, 10.0, len(df))
    if "Conductivity" not in df.columns:
      df["Conductivity"] = np.random.uniform(100, 1200, len(df))
    if "Turbidity" not in df.columns:
      df["Turbidity"] = np.random.uniform(1.0, 15.0, len(df))
    if "pH" not in df.columns:
      df["pH"] = np.random.uniform(6.5, 8.5, len(df))

    # Target column handling
    if "Condition" in df.columns:
      if set(df["Condition"].dropna().unique()).issubset({0, 1}):
        df["Condition"] = df["Condition"].map({1: "Normal", 0: "Polluted"})
    else:
      conditions = []
      for i in range(len(df)):
        if (
            df["pH"].iloc[i] < 6.0
            or df["pH"].iloc[i] > 8.5
            or df["Turbidity"].iloc[i] > 10.0
        ):
          conditions.append("Polluted")
        else:
          conditions.append("Normal")
      df["Condition"] = conditions

    df = df.fillna(df.median(numeric_only=True))

  else:
    np.random.seed(42)
    n_samples = 500

    ph = np.random.uniform(5.5, 9.0, n_samples)
    temp = np.random.uniform(20.0, 35.0, n_samples)
    turbidity = np.random.uniform(1.0, 15.0, n_samples)
    conductivity = np.random.uniform(100, 1200, n_samples)
    do = np.random.uniform(2.0, 10.0, n_samples)

    conditions = []
    for i in range(n_samples):
      if (
          do[i] < 4.0
          or ph[i] < 6.0
          or ph[i] > 8.5
          or turbidity[i] > 10.0
          or conductivity[i] > 900
      ):
        if do[i] < 3.0 or turbidity[i] > 12.0:
          conditions.append("Hazardous")
        else:
          conditions.append("Polluted")
      else:
        conditions.append("Normal")

    df = pd.DataFrame({
        "pH": ph,
        "Temperature": temp,
        "Turbidity": turbidity,
        "Conductivity": conductivity,
        "DO": do,
        "Condition": conditions,
    })
  return df


df = load_data(uploaded_file)

if uploaded_file is not None:
  st.sidebar.success("CSV File Processed & Prepared!")

# --- Step 3: Model Training ---
X = df[["pH", "Temperature", "Turbidity", "Conductivity", "DO"]]
y = df["Condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Auto-fill Parameter State Management ---
if "selected_sample" not in st.session_state:
  st.session_state.selected_sample = 0

st.sidebar.header("📊 Input / Live Sensor Data")

# Option to select a row from CSV to auto-fill sliders
if uploaded_file is not None:
  selected_row_idx = st.sidebar.selectbox(
      "Auto-fill from CSV Sample Row:",
      options=list(range(len(df))),
      index=st.session_state.selected_sample,
  )

  if selected_row_idx != st.session_state.selected_sample:
    st.session_state.selected_sample = selected_row_idx

  # Extract parameters from selected row
  default_ph = float(df.iloc[st.session_state.selected_sample]["pH"])
  default_temp = float(df.iloc[st.session_state.selected_sample]["Temperature"])
  default_turb = float(df.iloc[st.session_state.selected_sample]["Turbidity"])
  default_cond = float(
      df.iloc[st.session_state.selected_sample]["Conductivity"]
  )
  default_do = float(df.iloc[st.session_state.selected_sample]["DO"])
else:
  default_ph = 7.2
  default_temp = 26.5
  default_turb = 3.5
  default_cond = 450.0
  default_do = 6.8

# Sliders with dynamic default values from CSV
input_ph = st.sidebar.slider(
    "pH Level", 0.0, 14.0, value=min(max(default_ph, 0.0), 14.0)
)
input_temp = st.sidebar.slider(
    "Temperature (°C)", 0.0, 60.0, value=min(max(default_temp, 0.0), 60.0)
)
input_turb = st.sidebar.slider(
    "Turbidity (NTU)", 0.0, 30.0, value=min(max(default_turb, 0.0), 30.0)
)
input_cond = st.sidebar.slider(
    "Conductivity (µS/cm)",
    0.0,
    3000.0,
    value=min(max(default_cond, 0.0), 3000.0),
)
input_do = st.sidebar.slider(
    "Dissolved Oxygen - DO (mg/L)",
    0.0,
    20.0,
    value=min(max(default_do, 0.0), 20.0),
)

input_data = pd.DataFrame(
    [[input_ph, input_temp, input_turb, input_cond, input_do]],
    columns=["pH", "Temperature", "Turbidity", "Conductivity", "DO"],
)

# --- Output Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📌 Real-Time Prediction",
    "🔍 SHAP / Explainable AI",
    "📈 Historical & Data Trends",
])

# Tab 1: Live Status & Prediction
with tab1:
  st.subheader("Current Parameter Gauge & Status")

  col1, col2, col3, col4, col5 = st.columns(5)
  col1.metric("pH", f"{input_ph:.2f}")
  col2.metric("Temp (°C)", f"{input_temp:.2f}")
  col3.metric("Turbidity", f"{input_turb:.2f}")
  col4.metric("Conductivity", f"{input_cond:.2f}")
  col5.metric("DO (mg/L)", f"{input_do:.2f}")

  prediction = model.predict(input_data)[0]

  st.markdown("---")
  st.subheader("Model Prediction")
  if prediction == "Normal":
    st.success(f"✅ Predicted Water Condition: **{prediction}**")
  elif prediction == "Polluted":
    st.warning(f"⚠️️ Predicted Water Condition: **{prediction}**")
  else:
    st.error(f"🚨 Predicted Water Condition: **{prediction}**")

# Tab 2: Explainable AI (SHAP)
with tab2:
  st.subheader("💡 Explainable AI (SHAP Plot)")
  st.write(
      "This plot explains which water parameters influenced the AI decision"
      " for this specific sample."
  )

  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(input_data)

  class_idx = list(model.classes_).index(prediction)

  if isinstance(shap_values, list):
    p = shap.force_plot(
        explainer.expected_value[class_idx],
        shap_values[class_idx][0],
        input_data.iloc[0],
    )
  else:
    p = shap.force_plot(
        explainer.expected_value[class_idx],
        shap_values[0, :, class_idx],
        input_data.iloc[0],
    )

  st_shap(p, height=180)

# Tab 3: Data Trends
with tab3:
  st.subheader("Water Quality Dataset Overview")
  st.dataframe(df.head(20))

  st.subheader("Scatter Plot: Dissolved Oxygen vs Turbidity")
  fig_scatter = px.scatter(
      df,
      x="DO",
      y="Turbidity",
      color="Condition",
      title="Water Quality Clusters",
      color_discrete_map={
          "Normal": "green",
          "Polluted": "orange",
          "Hazardous": "red",
      },
  )
  st.plotly_chart(fig_scatter, use_container_width=True)