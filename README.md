# Breast Cancer Classification using Logistic Regression

A machine learning project that uses logistic regression to classify breast cancer tumors as malignant or benign based on diagnostic features.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements a logistic regression model to predict whether a breast cancer tumor is malignant (cancerous) or benign (non-cancerous). The model analyzes various features computed from digitized images of fine needle aspirate (FNA) of breast masses to make accurate predictions.

## 📊 Dataset

The project uses breast cancer diagnostic data containing features computed from cell nuclei characteristics. The dataset includes:

- Multiple diagnostic measurements
- Binary classification (Malignant/Benign)
- Feature importance analysis
- Test predictions and validation reports

## 📁 Project Structure

```
breast-cancer-logreg/
├── data/
│   ├── data.csv                      # Main dataset
│   ├── feature_importance.csv        # Feature importance scores
│   ├── final_tableau_data.csv        # Data prepared for Tableau visualization
│   └── test_predictions.csv          # Model predictions on test set
├── models/
│   ├── log_reg_model.pkl             # Trained logistic regression model
│   └── scaler.pkl                    # Feature scaler for preprocessing
├── notebooks/
│   └── data_exploration.ipynb        # Jupyter notebook for data analysis and modeling
├── reports/
│   └── validation_classification_report.csv  # Model validation metrics
├── visuals/
│   ├── Breast Cancer Classification Dashboard.png
│   ├── Breast_Cancer_Dashboard.twb   # Tableau workbook
│   ├── class_distribution.png        # Class distribution visualization
│   ├── cm_test.png                   # Confusion matrix (test set)
│   ├── cm_training.png               # Confusion matrix (training set)
│   ├── cm_validation.png             # Confusion matrix (validation set)
│   ├── full_correlation_heatmap.png  # Feature correlation heatmap
│   ├── roc_curve.png                 # ROC curve
│   └── top_10_correlations.png       # Top 10 feature correlations
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/OjPreshuz/breast-cancer-logreg.git
cd breast-cancer-logreg
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

Required libraries include:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- jupyter

## 💻 Usage

### Running the Jupyter Notebook

1. Start Jupyter Notebook:

```bash
jupyter notebook
```

2. Open `notebooks/data_exploration.ipynb` to:
   - Explore the dataset
   - Perform data preprocessing
   - Train the logistic regression model
   - Evaluate model performance
   - Generate visualizations

### Using the Trained Model

```python
import pickle
import pandas as pd

# Load the trained model and scaler
with open('models/log_reg_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions on new data
# X_new should be a DataFrame with the same features as training data
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
```

## 📈 Model Performance

The logistic regression model demonstrates strong performance in classifying breast cancer tumors:

- **Confusion Matrices**: Available for training, validation, and test sets
- **ROC Curve**: Shows the model's ability to distinguish between classes
- **Classification Report**: Detailed metrics including precision, recall, and F1-score
- **Feature Importance**: Identifies the most influential features for prediction

Check the `visuals/` directory for detailed performance visualizations.

## 📊 Visualizations

The project includes comprehensive visualizations:

1. **Class Distribution**: Shows the balance between malignant and benign cases
2. **Confusion Matrices**: Performance on training, validation, and test sets
3. **ROC Curve**: Model discrimination capability
4. **Correlation Heatmap**: Relationships between features
5. **Feature Importance**: Top contributing features
6. **Tableau Dashboard**: Interactive visualization of results

## 🛠️ Technologies Used

- **Python 3.x**: Primary programming language
- **Scikit-learn**: Machine learning model implementation
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Statsmodels**: Statistical analysis
- **Tableau**: Interactive dashboard creation
- **Jupyter Notebook**: Interactive development environment

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 📝 License

This project is open source and available for educational and research purposes.

## 👤 Author

**OjPreshuz**

- GitHub: [@OjPreshuz](https://github.com/OjPreshuz)

## 🙏 Acknowledgments

- Dataset source: Breast Cancer Wisconsin (Diagnostic) Dataset
- Inspired by the need for accurate and accessible cancer diagnosis tools

---

**Note**: This project is for educational purposes. Always consult healthcare professionals for medical diagnoses.
