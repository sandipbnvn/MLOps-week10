# Iris Detection with Governance

This project demonstrates machine learning model development with a focus on explainability and fairness using the classic Iris dataset. The implementation includes model training, SHAP-based explainability analysis, and fairness evaluation using Fairlearn.

## Project Overview

The project builds a Decision Tree classifier to predict Iris flower species and implements governance practices including:
- Model explainability using SHAP (SHapley Additive exPlanations)
- Fairness evaluation using Fairlearn
- Model artifacts management

## Dataset

The project uses the Iris dataset (`data/iris.csv`) which contains:
- **Features**: sepal_length, sepal_width, petal_length, petal_width
- **Target**: species (setosa, versicolor, virginica)
- **Additional attribute**: location (artificially introduced for fairness analysis)

## Project Structure

```
week9/
├── data/
│   └── iris.csv                    # Iris dataset
├── artifacts/
│   └── model.joblib               # Trained model artifacts
├── Iris detection with governance.ipynb  # Main notebook
└── README.md                      # This file
```

## Key Components

### 1. Model Training
- **Algorithm**: Decision Tree Classifier
- **Parameters**: max_depth=4, random_state=1
- **Performance**: 95.0% accuracy on test set
- **Train-Test Split**: 60/40 ratio with stratified sampling

### 2. Model Explainability (SHAP)
The project implements SHAP analysis to understand model predictions:
- **SHAP Waterfall Plots**: Show individual prediction explanations
- **Feature Importance**: Identifies which features contribute most to predictions
- **SHAP Values**: Quantify each feature's impact on model output

Key findings from SHAP analysis:
- For virginica classification, petal_length and petal_width features have the highest positive contributions
- SHAP values help understand feature importance for individual predictions

### 3. Fairness Analysis (Fairlearn)
Fairness evaluation is performed using an artificially introduced "location" attribute:
- **Sensitive Attribute**: location (randomly assigned with 70/30 distribution)
- **Metrics Evaluated**: accuracy, precision, recall
- **Fairness Assessment**: Model performance is compared across different location groups

Fairness results:
- Overall accuracy: 93.3%
- Location group 0: 90.2% accuracy
- Location group 1: 100% accuracy
- Shows potential bias across location groups

## Dependencies

```python
pandas
numpy
scikit-learn
shap
fairlearn
matplotlib
joblib
```

## Installation

1. Clone or download the project
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn shap fairlearn matplotlib joblib
   ```

## Usage

1. **Run the Jupyter Notebook**: Open `Iris detection with governance.ipynb`
2. **Execute cells sequentially** to:
   - Load and explore the data
   - Train the Decision Tree model
   - Generate SHAP explanations
   - Perform fairness analysis

## Key Results

### Model Performance
- **Accuracy**: 95.0% (without location feature)
- **Accuracy with location**: 91.7% (slight decrease when location is included)

### Explainability Insights
- Petal measurements (length and width) are the most important features for virginica classification
- SHAP values provide quantitative measure of each feature's contribution

### Fairness Assessment
- Model shows different performance across location groups
- Demonstrates the importance of fairness evaluation in ML pipelines
- Highlights potential bias that could occur with geographic or demographic attributes

## Governance Practices Demonstrated

1. **Explainability**: Using SHAP to understand model decisions
2. **Fairness**: Evaluating model performance across sensitive attributes
3. **Reproducibility**: Setting random seeds for consistent results
4. **Artifact Management**: Saving trained models for deployment

## Next Steps

- Implement bias mitigation techniques
- Add more comprehensive fairness metrics
- Create automated model monitoring pipeline
- Implement model versioning and experiment tracking

## License

This project is for educational purposes and demonstrates MLOps governance practices.
