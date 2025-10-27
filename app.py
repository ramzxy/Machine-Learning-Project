import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import stats

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

# Feature Separation Analysis (from statistics.py)
"""
## :material/analytics: Feature Distribution Overlap
Compare the distribution overlap between diabetic and non-diabetic patients
"""

# Prepare data for analysis
positive_data = dataset[dataset['class'] == 'tested_positive']
negative_data = dataset[dataset['class'] == 'tested_negative']

# Calculate separation scores (same as statistics.py)
features = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']
variance_analysis = []

for feature in features:
    pos_mean = positive_data[feature].mean()
    neg_mean = negative_data[feature].mean()
    diff = abs(pos_mean - neg_mean)
    pos_std = positive_data[feature].std()
    neg_std = negative_data[feature].std()
    
    variance_analysis.append({
        'feature': feature,
        'mean_diff': round(diff, 2),
        'positive_mean': round(pos_mean, 2),
        'negative_mean': round(neg_mean, 2),
        'positive_std': round(pos_std, 2),
        'negative_std': round(neg_std, 2),
        'separation_score': round(diff / ((pos_std + neg_std) / 2), 3)
    })

df_variance = pd.DataFrame(variance_analysis).sort_values('separation_score', ascending=False)

feature_names_full = {
    'plas': 'Glucose', 
    'mass': 'BMI', 
    'age': 'Age',
    'preg': 'Pregnancies', 
    'pedi': 'Pedigree',
    'pres': 'Blood Pressure',
    'skin': 'Skin Thickness',
    'insu': 'Insulin'
}

# Feature selector
st.write("**Select a feature to view its distribution overlap**")
st.caption("The less overlap between distributions, the better the feature separates diabetic from non-diabetic patients")

selected_feature = st.selectbox(
    "Choose feature",
    ['plas', 'mass', 'age', 'pedi', 'preg', 'pres', 'skin', 'insu'],
    format_func=lambda x: feature_names_full[x],
    label_visibility="collapsed"
)

# Get stats for selected feature
feature_data = df_variance[df_variance['feature'] == selected_feature].iloc[0]
feature_name = feature_names_full[selected_feature]
sep_score = feature_data['separation_score']
pos_mean = feature_data['positive_mean']
neg_mean = feature_data['negative_mean']
pos_std = feature_data['positive_std']
neg_std = feature_data['negative_std']

# Display statistics
stat_cols = st.columns(5)
with stat_cols[0]:
    st.metric("Separation Score", f"{sep_score:.3f}", help="Higher = better separation")
with stat_cols[1]:
    st.metric("Diabetic Mean", f"{pos_mean:.2f}")
with stat_cols[2]:
    st.metric("Non-Diabetic Mean", f"{neg_mean:.2f}")
with stat_cols[3]:
    st.metric("Mean Difference", f"{abs(pos_mean - neg_mean):.2f}")
with stat_cols[4]:
    st.metric("Std Dev Avg", f"{((pos_std + neg_std) / 2):.2f}")

""

# Create distribution plot for selected feature
fig = go.Figure()

# Non-diabetic histogram
fig.add_trace(go.Histogram(
    x=negative_data[selected_feature],
    name='Non-Diabetic',
    opacity=0.6,
    marker_color='#4caf50',
    histnorm='probability density',
    nbinsx=30
))

# Diabetic histogram
fig.add_trace(go.Histogram(
    x=positive_data[selected_feature],
    name='Diabetic',
    opacity=0.6,
    marker_color='#f44336',
    histnorm='probability density',
    nbinsx=30
))

# Add normal distribution curves
x_range = np.linspace(
    min(dataset[selected_feature].min(), neg_mean - 3*neg_std, pos_mean - 3*pos_std),
    max(dataset[selected_feature].max(), neg_mean + 3*neg_std, pos_mean + 3*pos_std),
    200
)

# Non-diabetic normal curve
neg_curve = stats.norm.pdf(x_range, neg_mean, neg_std)
fig.add_trace(go.Scatter(
    x=x_range,
    y=neg_curve,
    name='Non-Diabetic (Normal)',
    line=dict(color='darkgreen', width=3),
    mode='lines'
))

# Diabetic normal curve
pos_curve = stats.norm.pdf(x_range, pos_mean, pos_std)
fig.add_trace(go.Scatter(
    x=x_range,
    y=pos_curve,
    name='Diabetic (Normal)',
    line=dict(color='darkred', width=3),
    mode='lines'
))

fig.update_layout(
    title=f"{feature_name} Distribution - Diabetic vs Non-Diabetic",
    xaxis_title=feature_name,
    yaxis_title="Density",
    height=500,
    barmode='overlay',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig, use_container_width=True)

""
""

# Feature Correlations
"""
## :material/grid_on: Feature Correlations
Understanding how features relate to each other and to diabetes
"""

st.write("**What are correlations?** Correlations measure how strongly features are related (range: -1 to +1)")
st.caption("Positive correlation: both increase together | Negative correlation: one increases, other decreases | Zero: no relationship")

# Prepare correlation data
corr_data = dataset.copy()
corr_data['Diabetes'] = (corr_data['class'] == 'tested_positive').astype(int)

# Select features for correlation
corr_features = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age', 'Diabetes']
correlation_matrix = corr_data[corr_features].corr()

# Rename for display
feature_display_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                         'Insulin', 'BMI', 'Pedigree', 'Age', 'Diabetes']

# Create heatmap
fig = px.imshow(
    correlation_matrix,
    labels=dict(x="Feature", y="Feature", color="Correlation"),
    x=feature_display_names,
    y=feature_display_names,
    color_continuous_scale='RdBu_r',
    aspect='auto',
    text_auto='.2f',
    zmin=-1,
    zmax=1
)

fig.update_layout(
    title="Feature Correlation Heatmap",
    height=600,
    width=800
)

st.plotly_chart(fig, use_container_width=True)

""

# Show correlations with diabetes
st.write("**How Each Feature Correlates with Diabetes**")
st.caption("Higher absolute values = stronger relationship with diabetes risk")

diabetes_corr = correlation_matrix['Diabetes'].drop('Diabetes').sort_values(ascending=False)

corr_display = pd.DataFrame({
    'Feature': feature_display_names[:-1],  # Exclude 'Diabetes' itself
    'Correlation': diabetes_corr.values,
    'Strength': diabetes_corr.abs().values
})

# Create bar chart for correlations with diabetes
fig = go.Figure()

# Color bars based on positive/negative
colors = ['#f44336' if x > 0 else '#2196f3' for x in corr_display['Correlation']]

fig.add_trace(go.Bar(
    x=corr_display['Correlation'],
    y=corr_display['Feature'],
    orientation='h',
    marker=dict(
        color=colors,
        line=dict(color='black', width=0.5)
    ),
    text=corr_display['Correlation'].round(3),
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>'
))

fig.update_layout(
    title="Correlation with Diabetes Risk",
    xaxis_title="Correlation Coefficient",
    yaxis_title="",
    height=450,
    xaxis=dict(range=[-0.1, 0.55], zeroline=True, zerolinewidth=2, zerolinecolor='black'),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)



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
