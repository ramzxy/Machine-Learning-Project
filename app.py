import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        color: #ff7f0e;
        margin-top: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .negative {
        background-color: #ccffcc;
        border-left: 5px solid #00ff00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('diabetes_knn_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('dataset_37_diabetes.csv')
    # Apply same cleaning as training
    df = df[(df['plas'] > 0) & (df['pres'] > 0) & (df['mass'] > 0)]
    return df

# Load resources
try:
    model, scaler = load_model()
    dataset = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio("Go to", ["🔮 Make Prediction", "📊 Model Performance", "📈 Data Insights"])

# Main title
st.markdown('<p class="main-header">🏥 Diabetes Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

# PAGE 1: PREDICTION
if page == "🔮 Make Prediction":
    st.markdown('<p class="sub-header">Enter Patient Information</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Medical Measurements")
        
        plas = st.slider(
            "Plasma Glucose Concentration (mg/dL)",
            min_value=50,
            max_value=200,
            value=120,
            help="Normal fasting glucose: 70-100 mg/dL"
        )
        
        mass = st.slider(
            "Body Mass Index (BMI)",
            min_value=15.0,
            max_value=60.0,
            value=25.0,
            step=0.1,
            help="Normal BMI: 18.5-24.9"
        )
        
        age = st.slider(
            "Age (years)",
            min_value=21,
            max_value=80,
            value=30
        )
    
    with col2:
        st.subheader("👤 Patient History")
        
        preg = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=15,
            value=0,
            help="Total number of pregnancies"
        )
        
        pedi = st.slider(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=2.5,
            value=0.5,
            step=0.01,
            help="Genetic diabetes likelihood (0-2.5)"
        )
    
    # Display input summary
    st.markdown("---")
    st.subheader("📝 Input Summary")
    
    input_data = pd.DataFrame({
        'Feature': ['Glucose', 'BMI', 'Age', 'Pregnancies', 'Pedigree'],
        'Value': [plas, mass, age, preg, pedi],
        'Status': [
            '🔴 High' if plas > 126 else '🟡 Prediabetic' if plas > 100 else '🟢 Normal',
            '🔴 Obese' if mass >= 30 else '🟡 Overweight' if mass >= 25 else '🟢 Normal',
            '🟡 Older' if age >= 45 else '🟢 Younger',
            '🟡 Multiple' if preg > 3 else '🟢 Few/None',
            '🔴 High' if pedi > 0.5 else '🟢 Low'
        ]
    })
    
    st.dataframe(input_data, use_container_width=True, hide_index=True)
    
    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True):
        # Prepare input in the SAME ORDER as training: preg, plas, mass, pedi, age
        input_features = pd.DataFrame([[preg, plas, mass, pedi, age]], 
                                      columns=['preg', 'plas', 'mass', 'pedi', 'age'])
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display result
        st.markdown("---")
        st.markdown('<p class="sub-header">🎯 Prediction Result</p>', unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-box positive">
                    <h2 style="color: #cc0000;">⚠️ HIGH RISK - Diabetes Detected</h2>
                    <p style="font-size: 20px;">Confidence: {probability[1]*100:.1f}%</p>
                    <p>The patient shows indicators consistent with diabetes. Medical consultation recommended.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-box negative">
                    <h2 style="color: #00aa00;">✅ LOW RISK - No Diabetes Detected</h2>
                    <p style="font-size: 20px;">Confidence: {probability[0]*100:.1f}%</p>
                    <p>The patient shows indicators consistent with no diabetes. Continue healthy lifestyle.</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Show probability gauge
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability[1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diabetes Risk (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if probability[1] > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance for this prediction
            st.subheader("📊 Risk Factors")
            
            feature_importance = pd.DataFrame({
                'Factor': ['Glucose', 'BMI', 'Age', 'Pregnancies', 'Pedigree'],
                'Your Value': [plas, mass, age, preg, pedi],
                'Risk Level': [
                    min(100, (plas / 200) * 100),
                    min(100, (mass / 60) * 100),
                    min(100, (age / 80) * 100),
                    min(100, (preg / 15) * 100),
                    min(100, (pedi / 2.5) * 100)
                ]
            })
            
            fig = px.bar(
                feature_importance,
                x='Risk Level',
                y='Factor',
                orientation='h',
                text='Your Value',
                color='Risk Level',
                color_continuous_scale=['green', 'yellow', 'red']
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

# PAGE 2: MODEL PERFORMANCE
elif page == "📊 Model Performance":
    st.markdown('<p class="sub-header">Model Performance Metrics</p>', unsafe_allow_html=True)
    
    # Calculate metrics
    from sklearn.model_selection import train_test_split
    
    dataset_clean = dataset.copy()
    dataset_clean.drop(columns=['pres', 'skin', 'insu'], inplace=True)
    dataset_clean['class_numeric'] = (dataset_clean['class'] == 'tested_positive').astype(int)
    
    # Features in SAME ORDER as training: preg, plas, mass, pedi, age
    features = ['preg', 'plas', 'mass', 'pedi', 'age']
    X = dataset_clean[features]
    y = dataset_clean['class_numeric']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    accuracy = (y_pred == y_test).mean()
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Accuracy</h3>
                <h1>{accuracy*100:.2f}%</h1>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Total Samples</h3>
                <h1>{len(dataset_clean)}</h1>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3>🔬 Test Samples</h3>
                <h1>{len(y_test)}</h1>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h3>🎲 Features Used</h3>
                <h1>5</h1>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['No Diabetes', 'Diabetes'],
            y=['No Diabetes', 'Diabetes'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            colorscale='Blues'
        ))
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.info(f"""
        **Confusion Matrix Interpretation:**
        - ✅ True Negatives: {cm[0][0]} (Correctly predicted No Diabetes)
        - ✅ True Positives: {cm[1][1]} (Correctly predicted Diabetes)
        - ❌ False Positives: {cm[0][1]} (Incorrectly predicted Diabetes)
        - ❌ False Negatives: {cm[1][0]} (Missed Diabetes cases)
        """)
    
    with col2:
        st.subheader("📈 Classification Report")
        
        # Get classification report as dict
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        report_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score'],
            'No Diabetes': [
                precision_score(y_test, y_pred, pos_label=0),
                recall_score(y_test, y_pred, pos_label=0),
                f1_score(y_test, y_pred, pos_label=0)
            ],
            'Diabetes': [
                precision_score(y_test, y_pred, pos_label=1),
                recall_score(y_test, y_pred, pos_label=1),
                f1_score(y_test, y_pred, pos_label=1)
            ]
        })
        
        st.dataframe(report_df.set_index('Metric').style.format("{:.3f}"), use_container_width=True)
        
        # Bar chart of metrics
        fig = go.Figure()
        fig.add_trace(go.Bar(name='No Diabetes', x=report_df['Metric'], y=report_df['No Diabetes']))
        fig.add_trace(go.Bar(name='Diabetes', x=report_df['Metric'], y=report_df['Diabetes']))
        fig.update_layout(barmode='group', height=300, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance (based on separation scores from statistics)
    st.subheader("⭐ Feature Importance")
    
    feature_stats = pd.DataFrame({
        'Feature': ['Glucose', 'BMI', 'Age', 'Pregnancies', 'Pedigree'],
        'Separation Score': [1.077, 0.647, 0.519, 0.464, 0.360],
        'Correlation': [0.495, 0.314, 0.238, 0.222, 0.174]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Separation Score',
        x=feature_stats['Feature'],
        y=feature_stats['Separation Score'],
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name='Correlation',
        x=feature_stats['Feature'],
        y=feature_stats['Correlation'],
        marker_color='lightcoral'
    ))
    fig.update_layout(barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Feature Importance Explained:**
    - **Separation Score**: How well the feature separates diabetes vs non-diabetes cases (higher is better)
    - **Correlation**: How strongly the feature correlates with diabetes (0-1 scale)
    - **Glucose** is the most important feature, followed by **BMI** and **Age**
    """)

# PAGE 3: DATA INSIGHTS
elif page == "📈 Data Insights":
    st.markdown('<p class="sub-header">Dataset Analysis & Insights</p>', unsafe_allow_html=True)
    
    # Class distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🍰 Class Distribution")
        class_counts = dataset['class'].value_counts()
        
        fig = px.pie(
            values=class_counts.values,
            names=['No Diabetes', 'Diabetes'],
            color_discrete_sequence=['lightgreen', 'lightcoral']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label+value')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Dataset Statistics")
        st.metric("Total Patients", len(dataset))
        st.metric("Diabetes Cases", len(dataset[dataset['class'] == 'tested_positive']))
        st.metric("Non-Diabetes Cases", len(dataset[dataset['class'] == 'tested_negative']))
        st.metric("Diabetes Rate", f"{len(dataset[dataset['class'] == 'tested_positive'])/len(dataset)*100:.1f}%")
    
    st.markdown("---")
    
    # Feature distributions
    st.subheader("📈 Feature Distributions by Class")
    
    feature_select = st.selectbox(
        "Select Feature to Visualize",
        ['plas', 'mass', 'age', 'preg', 'pedi']
    )
    
    feature_names = {
        'plas': 'Glucose (mg/dL)',
        'mass': 'BMI',
        'age': 'Age (years)',
        'preg': 'Pregnancies',
        'pedi': 'Pedigree Function'
    }
    
    fig = px.histogram(
        dataset,
        x=feature_select,
        color='class',
        barmode='overlay',
        labels={'class': 'Diagnosis', feature_select: feature_names[feature_select]},
        color_discrete_map={'tested_negative': 'lightgreen', 'tested_positive': 'lightcoral'},
        opacity=0.7
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    
    dataset_numeric = dataset[['preg', 'plas', 'mass', 'pedi', 'age']].copy()
    corr_matrix = dataset_numeric.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Age vs Glucose scatter
    st.subheader("🎯 Age vs Glucose (Colored by Diagnosis)")
    
    fig = px.scatter(
        dataset,
        x='age',
        y='plas',
        color='class',
        size='mass',
        hover_data=['preg', 'pedi'],
        labels={'age': 'Age (years)', 'plas': 'Glucose (mg/dL)', 'class': 'Diagnosis'},
        color_discrete_map={'tested_negative': 'lightgreen', 'tested_positive': 'lightcoral'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>🏥 Diabetes Prediction System | Built with KNN Machine Learning | Accuracy: 81.38%</p>
        <p>⚠️ This is a predictive tool and should not replace professional medical advice.</p>
    </div>
""", unsafe_allow_html=True)

