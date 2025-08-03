# 🛒 BigMart Sales Prediction Dashboard

An interactive Streamlit dashboard for exploring the BigMart Sales Prediction project, analyzing model performance, and making predictions.

## 🚀 Features

### 📊 Overview Page
- Project summary and key metrics
- Quick statistics about training and test data
- Model performance summary

### 📈 Data Exploration Page
- Interactive feature analysis
- Distribution plots for numerical and categorical features
- Correlation analysis with target variable
- Dynamic visualizations using Plotly

### 🤖 Model Performance Page
- Comprehensive model comparison charts
- Detailed performance metrics
- Best model identification and analysis
- Interactive performance visualizations

### 🔮 Make Predictions Page
- **Test Data Predictions**: Generate predictions for all test samples
- **Single Item Prediction**: Interactive form for predicting individual items
- **Custom Data Upload**: Upload CSV files for batch predictions
- Download predictions as CSV files

### 📋 Data Summary Page
- Detailed dataset information
- Missing values analysis
- Data types overview
- Comprehensive data statistics

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure models are trained:**
   ```bash
   python scripts/train.py
   ```

## 🎯 Usage

1. **Run the dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open in browser:**
   The dashboard will automatically open in your default browser at `http://localhost:8501`

## 📱 Dashboard Navigation

### Sidebar Navigation
- **📊 Overview**: Project summary and key metrics
- **📈 Data Exploration**: Interactive data analysis
- **🤖 Model Performance**: Model comparison and analysis
- **🔮 Make Predictions**: Prediction interface
- **📋 Data Summary**: Detailed data information

### Key Features

#### Data Exploration
- Select numerical or categorical features for analysis
- Interactive histograms, box plots, and bar charts
- Correlation analysis with target variable
- Real-time visualizations

#### Model Performance
- Multi-metric model comparison (RMSE, R², MAE, CV-RMSE)
- Interactive performance charts
- Best model identification
- Detailed performance tables

#### Predictions
- **Test Data**: Generate predictions for all test samples
- **Single Item**: Interactive form with all feature inputs
- **Custom Upload**: Upload CSV files for batch predictions
- Download results as CSV files

## 🎨 Customization

### Styling
The dashboard includes custom CSS for:
- Professional color scheme
- Responsive layout
- Metric cards with borders
- Custom header styling

### Adding New Features
To add new features to the dashboard:

1. **Add new page function** in `streamlit_app.py`
2. **Update navigation** in the sidebar
3. **Add page routing** in the main function

### Example:
```python
def show_new_feature():
    st.header("New Feature")
    # Your code here

# In main():
elif page == "New Feature":
    show_new_feature()
```

## 📊 Data Requirements

The dashboard expects:
- Trained models in `models/` directory
- Training and test data in `data/raw/` directory
- Model comparison results from training

## 🔧 Troubleshooting

### Common Issues

1. **Model not found error:**
   - Ensure models are trained: `python scripts/train.py`
   - Check model files exist in `models/` directory

2. **Import errors:**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python path includes project root

3. **Data loading errors:**
   - Verify data files exist in `data/raw/` directory
   - Check file permissions

### Performance Tips

1. **Large datasets:**
   - Use data sampling for exploration
   - Implement caching for expensive operations

2. **Memory usage:**
   - Load data only when needed
   - Clear cache periodically

## 📈 Future Enhancements

Potential improvements:
- Real-time model retraining
- Advanced feature importance visualization
- Model interpretability tools
- Export functionality for reports
- User authentication
- Multi-user support

## 🤝 Contributing

To contribute to the dashboard:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This dashboard is part of the BigMart Sales Prediction project.

---

**Happy exploring! 🎉** 