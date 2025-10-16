import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon=":material/health_and_safety:",
    layout="wide",
)

# Load model and data
@st.cache_resource
def load_model():
    model = joblib.load('diabetes_knn_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('dataset_37_diabetes.csv')
    df = df[(df['plas'] > 0) & (df['pres'] > 0) & (df['mass'] > 0)]
    return df

try:
    model, scaler = load_model()
    dataset = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Title
"""
# :material/health_and_safety: Diabetes Risk Prediction

AI-powered health assessment using K-Nearest Neighbors algorithm.  
**Model Accuracy: 81.38%** | 724 training samples
"""

""

# Main layout
cols = st.columns([1, 2])

# LEFT PANEL - Input controls
left_panel = cols[0].container(border=True, height=600, vertical_alignment="top")

with left_panel:
    st.subheader("Patient Information")
    
    plas = st.slider(
        ":material/bloodtype: Glucose (mg/dL)",
        50, 200, 120,
        help="Fasting: Normal <100, Prediabetic 100-125, Diabetic ≥126"
    )
    
    mass = st.slider(
        ":material/monitor_weight: BMI",
        15.0, 60.0, 25.0, 0.1,
        help="Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30"
    )
    
    age = st.slider(
        ":material/person: Age (years)",
        21, 80, 30
    )
    
    preg = st.number_input(
        ":material/family_restroom: Pregnancies",
        0, 15, 0,
        help="Total number of pregnancies"
    )
    
    pedi = st.slider(
        ":material/genetics: Pedigree Function",
        0.0, 2.5, 0.5, 0.01,
        help="Genetic diabetes likelihood"
    )
    
    ""
    predict_button = st.button(
        ":material/query_stats: Predict Risk",
        type="primary",
        use_container_width=True
    )

# RIGHT PANEL - Results
right_panel = cols[1].container(border=True, height=650, vertical_alignment="top")

if predict_button:
    # Prepare input
    input_features = pd.DataFrame([[preg, plas, mass, pedi, age]], 
                                  columns=['preg', 'plas', 'mass', 'pedi', 'age'])
    input_scaled = scaler.transform(input_features)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    with right_panel:
        st.subheader("Prediction Results")
        
        # Main result
        if prediction == 1:
            st.error(f"""
            ### :material/warning: HIGH RISK - Diabetes Detected
            **Confidence: {probability[1]*100:.1f}%**
            
            The patient shows indicators consistent with diabetes.  
            :material/medical_services: Medical consultation recommended.
            """)
        else:
            st.success(f"""
            ### :material/check_circle: LOW RISK - No Diabetes Detected
            **Confidence: {probability[0]*100:.1f}%**
            
            The patient shows indicators consistent with no diabetes.  
            :material/favorite: Continue healthy lifestyle.
            """)
        
        ""
        
        # Risk metrics
        st.write("**Risk Factor Analysis**")
        metric_cols = st.columns(5)
        
        with metric_cols[0]:
            glucose_delta = plas - 100
            st.metric(
                "Glucose",
                f"{plas}",
                delta=f"{glucose_delta:+.0f}",
                delta_color="inverse"
            )
        
        with metric_cols[1]:
            bmi_delta = mass - 25
            st.metric(
                "BMI",
                f"{mass:.1f}",
                delta=f"{bmi_delta:+.1f}",
                delta_color="inverse"
            )
        
        with metric_cols[2]:
            st.metric("Age", f"{age}")
        
        with metric_cols[3]:
            st.metric("Pregnancies", f"{preg}")
        
        with metric_cols[4]:
            st.metric("Pedigree", f"{pedi:.2f}")
        
        ""
        
        # Confidence gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Diabetes Risk Level", 'font': {'size': 20}},
            number={'suffix': "%", 'font': {'size': 40}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "darkred" if probability[1] > 0.5 else "darkgreen", 'thickness': 0.8},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 30], 'color': '#a5d6a7'},
                    {'range': [30, 70], 'color': '#fff59d'},
                    {'range': [70, 100], 'color': '#ef9a9a'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

else:
    with right_panel:
        st.info("Enter patient information and click **Predict Risk** to see results.", 
                icon=":material/info:")

""
""

# Model Performance Section
"""
## :material/analytics: Model Performance
"""

# Calculate metrics
dataset_clean = dataset.copy()
dataset_clean.drop(columns=['pres', 'skin', 'insu'], inplace=True)
dataset_clean['class_numeric'] = (dataset_clean['class'] == 'tested_positive').astype(int)

features = ['preg', 'plas', 'mass', 'pedi', 'age']
X = dataset_clean[features]
y = dataset_clean['class_numeric']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

accuracy = (y_pred == y_test).mean()
cm = confusion_matrix(y_test, y_pred)

perf_cols = st.columns(3)

# Accuracy metrics
with perf_cols[0].container(border=True, height=400, vertical_alignment="center"):
    st.metric("Model Accuracy", f"{accuracy*100:.2f}%", help="Accuracy on test set")
    st.metric("Training Samples", "724", help="After data cleaning")
    st.metric("Test Samples", f"{len(y_test)}")

# Confusion Matrix
with perf_cols[1].container(border=True, height=400):
    st.write("**Confusion Matrix**")
    
    cm_data = pd.DataFrame({
        'Predicted No': [cm[0][0], cm[1][0]],
        'Predicted Yes': [cm[0][1], cm[1][1]]
    }, index=['Actual No', 'Actual Yes'])
    
    st.dataframe(cm_data, use_container_width=True)
    
    st.caption(f"✓ Correct: {cm[0][0] + cm[1][1]} | ✗ Incorrect: {cm[0][1] + cm[1][0]}")

# Feature Importance
with perf_cols[2].container(border=True, height=400):
    st.write("**Feature Importance**")
    
    feature_importance = pd.DataFrame({
        'Feature': ['Glucose', 'BMI', 'Age', 'Pregnancies', 'Pedigree'],
        'Importance': [1.077, 0.647, 0.519, 0.464, 0.360]
    })
    
    chart = alt.Chart(feature_importance).mark_bar().encode(
        x=alt.X('Importance:Q', title='Separation Score'),
        y=alt.Y('Feature:N', sort='-x', title=None),
        color=alt.Color('Importance:Q', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=['Feature', alt.Tooltip('Importance:Q', format='.3f')]
    ).properties(height=280)
    
    st.altair_chart(chart, use_container_width=True)

""
""

# Data Insights Section
"""
## :material/insights: Dataset Insights
"""

insight_cols = st.columns(2)

# Class Distribution
with insight_cols[0].container(border=True):
    st.write("**Class Distribution**")
    
    class_counts = dataset['class'].value_counts()
    class_data = pd.DataFrame({
        'Diagnosis': ['No Diabetes', 'Diabetes'],
        'Count': [class_counts['tested_negative'], class_counts['tested_positive']],
        'Percentage': [
            class_counts['tested_negative'] / len(dataset) * 100,
            class_counts['tested_positive'] / len(dataset) * 100
        ]
    })
    
    chart = alt.Chart(class_data).mark_arc(innerRadius=50).encode(
        theta=alt.Theta('Count:Q'),
        color=alt.Color('Diagnosis:N', 
                       scale=alt.Scale(domain=['No Diabetes', 'Diabetes'],
                                     range=['#4caf50', '#f44336'])),
        tooltip=['Diagnosis', 'Count', alt.Tooltip('Percentage:Q', format='.1f')]
    ).properties(height=350)
    
    st.altair_chart(chart, use_container_width=True)

# Feature Distribution
with insight_cols[1].container(border=True):
    st.write("**Feature Distribution**")
    
    feature_select = st.selectbox(
        "Select feature to visualize",
        ['plas', 'mass', 'age', 'preg', 'pedi'],
        format_func=lambda x: {
            'plas': 'Glucose', 
            'mass': 'BMI', 
            'age': 'Age',
            'preg': 'Pregnancies', 
            'pedi': 'Pedigree'
        }[x],
        label_visibility="collapsed"
    )
    
    hist_data = dataset[[feature_select, 'class']].copy()
    hist_data['Diagnosis'] = hist_data['class'].map({
        'tested_negative': 'No Diabetes',
        'tested_positive': 'Diabetes'
    })
    
    chart = alt.Chart(hist_data).mark_bar(opacity=0.7).encode(
        x=alt.X(f'{feature_select}:Q', bin=alt.Bin(maxbins=30)),
        y=alt.Y('count()', stack=None),
        color=alt.Color('Diagnosis:N',
                       scale=alt.Scale(domain=['No Diabetes', 'Diabetes'],
                                     range=['#4caf50', '#f44336'])),
        tooltip=['Diagnosis', 'count()']
    ).properties(height=350)
    
    st.altair_chart(chart, use_container_width=True)

""
""

# Raw Data
with st.expander("## :material/table: View Raw Dataset"):
    st.dataframe(dataset, use_container_width=True, height=400)

""

# Footer
st.caption("""
⚠️ **Disclaimer:** This tool is for educational purposes only.  
Always consult healthcare professionals for medical advice and diagnosis.
""")
