import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))

# Import project modules
from src.data.load_data import load_data
from src.models.predict import load_best_model, make_predictions
from src.utils.utils import load_object
from src.config.paths import SAVED_MODELS_DIR, METRICS_DIR, PLOTS_DIR

# Set page config
st.set_page_config(
    page_title="BigMart Sales Prediction Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def load_project_data():
    """Load all project data and models"""
    try:
        # Load data
        train_df, test_df = load_data()
        
        # Load models and results
        comparison_results = load_object(str(SAVED_MODELS_DIR / 'model_comparison_results.pkl'))
        encoders = load_object(str(SAVED_MODELS_DIR / 'preprocessing_encoders.pkl'))
        
        # Load evaluation results if available
        val_results_path = METRICS_DIR / 'validation_evaluation_results.csv'
        test_results_path = METRICS_DIR / 'test_evaluation_results.csv'
        
        val_results = None
        test_results = None
        
        if val_results_path.exists():
            val_results = pd.read_csv(val_results_path)
        if test_results_path.exists():
            test_results = pd.read_csv(test_results_path)
        
        return train_df, test_df, comparison_results, encoders, val_results, test_results
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🛒 BigMart Sales Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading project data..."):
        train_df, test_df, comparison_results, encoders, val_results, test_results = load_project_data()
    
    if train_df is None:
        st.error("Failed to load project data. Please ensure models have been trained.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Create navigation buttons - all always visible
    st.sidebar.markdown("### Available Pages:")
    
    # Use session state to track current page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Overview"
    
    # Create individual buttons for each page
    if st.sidebar.button("📊 Overview", use_container_width=True):
        st.session_state.current_page = "📊 Overview"
    
    if st.sidebar.button("📈 Data Exploration", use_container_width=True):
        st.session_state.current_page = "📈 Data Exploration"
    
    if st.sidebar.button("🤖 Model Performance", use_container_width=True):
        st.session_state.current_page = "🤖 Model Performance"
    
    if st.sidebar.button("🔮 Make Predictions", use_container_width=True):
        st.session_state.current_page = "🔮 Make Predictions"
    
    if st.sidebar.button("📋 Data Summary", use_container_width=True):
        st.session_state.current_page = "📋 Data Summary"
    
    # Show current page
    st.sidebar.markdown(f"**Current Page:** {st.session_state.current_page}")
    
    page = st.session_state.current_page
    
    if page == "📊 Overview":
        show_overview(train_df, test_df, comparison_results)
    elif page == "📈 Data Exploration":
        show_data_exploration(train_df, test_df)
    elif page == "🤖 Model Performance":
        show_model_performance(comparison_results, val_results, test_results)
    elif page == "🔮 Make Predictions":
        show_predictions(test_df, encoders)
    elif page == "📋 Data Summary":
        show_data_summary(train_df, test_df)

def show_overview(train_df, test_df, comparison_results):
    """Show project overview"""
    st.header("📊 Project Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", f"{len(train_df):,}")
    
    with col2:
        st.metric("Test Samples", f"{len(test_df):,}")
    
    with col3:
        best_model = comparison_results.iloc[0]['model_name']
        st.metric("Best Model", best_model)
    
    with col4:
        best_r2 = comparison_results.iloc[0]['val_r2']
        st.metric("Best R² Score", f"{best_r2:.3f}")
    
    # Project description
    st.markdown("""
    ### About the Project
    This dashboard provides insights into the BigMart Sales Prediction project, which aims to predict 
    the sales of products across different outlets. The project uses machine learning to forecast 
    sales based on various features including product characteristics, outlet information, and market conditions.
    
    ### Key Features
    - **Data Exploration**: Interactive visualizations of the dataset
    - **Model Performance**: Comparison of different machine learning models
    - **Predictions**: Make predictions on new data
    - **Real-time Analysis**: Dynamic insights and metrics
    """)
    
    # Quick stats
    st.subheader("📈 Quick Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Data Statistics:**")
        st.dataframe(train_df.describe())
    
    with col2:
        st.write("**Model Performance Summary:**")
        st.dataframe(comparison_results[['model_name', 'val_rmse', 'val_r2', 'val_mae']].round(3))

def show_data_exploration(train_df, test_df):
    """Show data exploration visualizations"""
    st.header("📈 Data Exploration")
    
    # Data overview
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Data Shape:**", train_df.shape)
        st.write("**Test Data Shape:**", test_df.shape)
    
    with col2:
        st.write("**Training Data Columns:**")
        st.write(list(train_df.columns))
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Select feature to analyze
    numerical_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
    
    feature_type = st.selectbox("Select feature type:", ["Numerical", "Categorical"])
    
    if feature_type == "Numerical":
        selected_feature = st.selectbox("Select numerical feature:", numerical_features)
        
        if selected_feature in train_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(train_df, x=selected_feature, title=f"Distribution of {selected_feature}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(train_df, y=selected_feature, title=f"Box Plot of {selected_feature}")
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        selected_feature = st.selectbox("Select categorical feature:", categorical_features)
        
        if selected_feature in train_df.columns:
            # Value counts
            value_counts = train_df[selected_feature].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"Distribution of {selected_feature}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"Proportion of {selected_feature}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    if 'Item_Outlet_Sales' in train_df.columns:
        st.subheader("Correlation Analysis")
        
        # Calculate correlation with target
        numerical_cols = train_df.select_dtypes(include=[np.number]).columns
        correlations = train_df[numerical_cols].corr()['Item_Outlet_Sales'].sort_values(ascending=False)
        
        # Plot correlations
        fig = px.bar(x=correlations.index, y=correlations.values, 
                    title="Feature Correlations with Target (Item_Outlet_Sales)")
        st.plotly_chart(fig, use_container_width=True)

def show_model_performance(comparison_results, val_results, test_results):
    """Show model performance analysis"""
    st.header("🤖 Model Performance")
    
    # Model comparison
    st.subheader("Model Comparison")
    
    # Create comparison chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Validation RMSE', 'Validation R²', 'Validation MAE', 'Cross-validation RMSE'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add traces
    fig.add_trace(
        go.Bar(x=comparison_results['model_name'], y=comparison_results['val_rmse'], 
               name='Validation RMSE', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=comparison_results['model_name'], y=comparison_results['val_r2'], 
               name='Validation R²', marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=comparison_results['model_name'], y=comparison_results['val_mae'], 
               name='Validation MAE', marker_color='lightcoral'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=comparison_results['model_name'], y=comparison_results['cv_rmse'], 
               name='CV RMSE', marker_color='lightyellow'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    st.dataframe(comparison_results.round(4))
    
    # Best model info
    best_model = comparison_results.iloc[0]
    st.subheader("🏆 Best Model")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", best_model['model_name'])
    
    with col2:
        st.metric("Validation RMSE", f"{best_model['val_rmse']:.2f}")
    
    with col3:
        st.metric("Validation R²", f"{best_model['val_r2']:.3f}")
    
    with col4:
        st.metric("Validation MAE", f"{best_model['val_mae']:.2f}")

def show_predictions(test_df, encoders):
    """Show prediction interface"""
    st.header("🔮 Make Predictions")
    
    # Load best model
    try:
        best_model, _, comparison_results = load_best_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Prediction options
    prediction_type = st.selectbox(
        "Choose prediction type:",
        ["Predict on Test Data", "Predict Single Item", "Upload Custom Data"]
    )
    
    if prediction_type == "Predict on Test Data":
        st.subheader("Test Data Predictions")
        
        if st.button("Generate Predictions"):
            with st.spinner("Making predictions..."):
                try:
                    predictions = make_predictions(test_df, best_model, encoders)
                    
                    # Create results dataframe
                    results_df = test_df[['Item_Identifier', 'Outlet_Identifier']].copy()
                    results_df['Predicted_Sales'] = predictions
                    
                    # Display results
                    st.success(f"✅ Generated {len(predictions)} predictions!")
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Prediction", f"${predictions.mean():.2f}")
                    with col2:
                        st.metric("Min Prediction", f"${predictions.min():.2f}")
                    with col3:
                        st.metric("Max Prediction", f"${predictions.max():.2f}")
                    
                    # Show predictions
                    st.subheader("Predictions Preview")
                    st.dataframe(results_df.head(20))
                    
                    # Download option
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv,
                        file_name="bigmart_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
    
    elif prediction_type == "Predict Single Item":
        st.subheader("Single Item Prediction")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                item_identifier = st.text_input("Item Identifier", "FDX07")
                item_weight = st.number_input("Item Weight", min_value=0.0, value=19.2)
                item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
                item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, value=0.0)
                item_type = st.selectbox("Item Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"])
                item_mrp = st.number_input("Item MRP", min_value=0.0, value=266.8884)
            
            with col2:
                outlet_identifier = st.selectbox("Outlet Identifier", ["OUT010", "OUT013", "OUT017", "OUT018", "OUT019", "OUT027", "OUT035", "OUT045", "OUT046", "OUT049"])
                outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1985, max_value=2009, value=2009)
                outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
                outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
                outlet_type = st.selectbox("Outlet Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
            
            submitted = st.form_submit_button("Predict Sales")
            
            if submitted:
                # Create input data
                input_data = pd.DataFrame([{
                    'Item_Identifier': item_identifier,
                    'Item_Weight': item_weight,
                    'Item_Fat_Content': item_fat_content,
                    'Item_Visibility': item_visibility,
                    'Item_Type': item_type,
                    'Item_MRP': item_mrp,
                    'Outlet_Identifier': outlet_identifier,
                    'Outlet_Establishment_Year': outlet_establishment_year,
                    'Outlet_Size': outlet_size,
                    'Outlet_Location_Type': outlet_location_type,
                    'Outlet_Type': outlet_type
                }])
                
                # Make prediction
                try:
                    prediction = make_predictions(input_data, best_model, encoders)[0]
                    
                    st.success("✅ Prediction generated!")
                    st.metric("Predicted Sales", f"${prediction:.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
    
    elif prediction_type == "Upload Custom Data":
        st.subheader("Upload Custom Data")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                custom_data = pd.read_csv(uploaded_file)
                st.write("Uploaded data preview:")
                st.dataframe(custom_data.head())
                
                if st.button("Make Predictions"):
                    with st.spinner("Making predictions..."):
                        predictions = make_predictions(custom_data, best_model, encoders)
                        
                        # Add predictions to dataframe
                        custom_data['Predicted_Sales'] = predictions
                        
                        st.success(f"✅ Generated {len(predictions)} predictions!")
                        st.dataframe(custom_data.head(20))
                        
                        # Download option
                        csv = custom_data.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name="custom_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

def show_data_summary(train_df, test_df):
    """Show detailed data summary"""
    st.header("📋 Data Summary")
    
    # Data info
    st.subheader("Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Data Info:**")
        # Create a proper info dataframe
        train_info = pd.DataFrame({
            'Column': train_df.columns,
            'Non-Null Count': train_df.count().values,
            'Data Type': train_df.dtypes.values,
            'Memory Usage': [train_df[col].memory_usage(deep=True) for col in train_df.columns]
        })
        st.dataframe(train_info)
        
        # Show basic stats
        st.write(f"**Shape:** {train_df.shape}")
        st.write(f"**Memory Usage:** {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        st.write("**Test Data Info:**")
        # Create a proper info dataframe
        test_info = pd.DataFrame({
            'Column': test_df.columns,
            'Non-Null Count': test_df.count().values,
            'Data Type': test_df.dtypes.values,
            'Memory Usage': [test_df[col].memory_usage(deep=True) for col in test_df.columns]
        })
        st.dataframe(test_info)
        
        # Show basic stats
        st.write(f"**Shape:** {test_df.shape}")
        st.write(f"**Memory Usage:** {test_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    st.subheader("Missing Values Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Data Missing Values:**")
        missing_train = train_df.isnull().sum()
        missing_train_pct = (missing_train / len(train_df)) * 100
        missing_df_train = pd.DataFrame({
            'Column': missing_train.index,
            'Missing Count': missing_train.values,
            'Missing Percentage': missing_train_pct.values
        })
        st.dataframe(missing_df_train[missing_df_train['Missing Count'] > 0])
    
    with col2:
        st.write("**Test Data Missing Values:**")
        missing_test = test_df.isnull().sum()
        missing_test_pct = (missing_test / len(test_df)) * 100
        missing_df_test = pd.DataFrame({
            'Column': missing_test.index,
            'Missing Count': missing_test.values,
            'Missing Percentage': missing_test_pct.values
        })
        st.dataframe(missing_df_test[missing_df_test['Missing Count'] > 0])
    
    # Data types
    st.subheader("Data Types")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Data Types:**")
        train_dtypes = train_df.dtypes.to_frame('Data Type')
        train_dtypes['Unique Values'] = [train_df[col].nunique() for col in train_df.columns]
        train_dtypes['Sample Values'] = [str(train_df[col].dropna().head(3).tolist()) for col in train_df.columns]
        st.dataframe(train_dtypes)
    
    with col2:
        st.write("**Test Data Types:**")
        test_dtypes = test_df.dtypes.to_frame('Data Type')
        test_dtypes['Unique Values'] = [test_df[col].nunique() for col in test_df.columns]
        test_dtypes['Sample Values'] = [str(test_df[col].dropna().head(3).tolist()) for col in test_df.columns]
        st.dataframe(test_dtypes)

if __name__ == "__main__":
    main() 