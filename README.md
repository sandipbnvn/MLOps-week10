# Fine-tuning Gemma 3 1B-IT for Iris Species Classification

**Week 10 Assignment - MLOps Course**  
**IITM BS Degree Program**  
**Student:** Sandip Biswas  
**Roll No:** 21F1002787

## Project Overview

This project demonstrates the fine-tuning of Google's Gemma 3 1B-IT model for Iris species classification. The implementation converts numerical flower measurements into categorical features (low, medium, high) and trains the model to classify Iris species: setosa, versicolor, and virginica.

## Week 10 Assignment Expected Outcomes for Evaluation

### 🔍 Schema Validation Checks/Testing on Generated Output

The implementation includes comprehensive validation mechanisms:

- **Output Format Validation**: All model predictions are validated against expected species names (setosa, versicolor, virginica)
- **Prediction Quality Assessment**: 
  - Validation rate tracking (1.000 for both base and fine-tuned models)
  - Empty output detection
  - Invalid prediction identification
- **Feature Schema Validation**: Input features are categorized using domain-informed boundaries based on KDE analysis
- **Data Integrity Checks**: Stratified sampling ensures balanced representation across all species

### 📊 Validating Improvements with Evaluation Metrics (Base vs Fine-tuned LLM)

#### Performance Comparison

| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| **Overall Accuracy** | 33.3% | 63.3% | +30.0% |
| **Setosa Accuracy** | 100% | 90% | -10% |
| **Versicolor Accuracy** | 0% | 100% | +100% |
| **Virginica Accuracy** | 0% | 0% | 0% |

#### Detailed Evaluation Results

**Base Model Performance:**
- Predicted only "setosa" for all test samples
- Complete failure to distinguish between versicolor and virginica
- Precision: 0.11 (macro avg), Recall: 0.33 (macro avg), F1-score: 0.17 (macro avg)

**Fine-tuned Model Performance:**
- Successfully learned to distinguish between setosa and versicolor
- Improved multi-class classification capability
- Precision: 0.45 (macro avg), Recall: 0.63 (macro avg), F1-score: 0.52 (macro avg)

#### Confusion Matrix Analysis

**Base Model:**
```
[[10  0  0]  <- All setosa correctly classified
 [10  0  0]  <- All versicolor misclassified as setosa  
 [10  0  0]] <- All virginica misclassified as setosa
```

**Fine-tuned Model:**
```
[[ 9  1  0]  <- 90% setosa accuracy
 [ 0 10  0]  <- 100% versicolor accuracy
 [ 2  8  0]] <- 0% virginica accuracy (confused with versicolor)
```

## Dataset and Methodology

### Dataset Specifications
- **Total Samples**: 150 (50 per species)
- **Training Set**: 90 samples (30 per species)
- **Test Set**: 30 samples (10 per species)
- **Evaluation Set**: 30 samples (10 per species)

### Feature Engineering
- Converted numerical measurements to categorical representations
- Domain-informed boundaries based on KDE analysis:
  - **Sepal Length**: Low (4.3-5.4), Medium (5.4-6.3), High (6.3-7.9)
  - **Sepal Width**: Low (2.0-2.7), Medium (2.7-3.3), High (3.3-4.4)
  - **Petal Length**: Low (1.0-2.1), Medium (3.0-5.0), High (5.0-7.0)
  - **Petal Width**: Low (0.1-0.6), Medium (1.0-1.6), High (1.6-2.5)

### Model Architecture and Training
- **Base Model**: Gemma 3 1B-IT
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Configuration**:
  - LoRA rank: 64, alpha: 32, dropout: 0.05
  - Learning rate: 2e-4
  - Epochs: 4
  - Batch size: 1 (with gradient accumulation: 8)

## Technical Implementation

### Key Technologies
- **Framework**: Transformers 4.50.0, PyTorch 2.6.0+cu124
- **Optimization**: BitsAndBytesConfig for memory efficiency
- **Training**: SFTTrainer with parameter-efficient fine-tuning
- **Evaluation**: Comprehensive metrics using scikit-learn

### Hardware Requirements
- CUDA-compatible GPU (used for model training)
- Sufficient memory for Gemma 3 1B model loading

## Results Summary

The fine-tuning process successfully improved the model's classification capabilities:

1. **Significant Overall Improvement**: 30% increase in accuracy
2. **Enhanced Multi-class Learning**: Base model could only predict one class, fine-tuned model distinguishes multiple classes
3. **Validation Framework**: 100% prediction validation rate ensures output quality
4. **Species-specific Performance**: Perfect versicolor classification, good setosa performance, challenges with virginica classification

## Files Structure

```
MLOps-week9/
├── fine-tune-gemma3-1b-it-iris-sandip.ipynb  # Main notebook
├── data/
│   ├── iris.csv                               # Original dataset
│   └── iris_train.jsonl                       # Training data in JSONL format
├── README.md                                  # This file
└── .gitignore                                # Git ignore file
```

## Future Improvements

1. **Enhanced Training Data**: Increase training samples for virginica classification
2. **Advanced Feature Engineering**: Explore additional categorical boundaries
3. **Model Architecture**: Experiment with different LoRA configurations
4. **Evaluation Metrics**: Implement additional validation frameworks

## Conclusion

This project demonstrates successful fine-tuning of a large language model for domain-specific classification tasks. The implementation includes robust validation mechanisms and comprehensive evaluation metrics, showing clear improvements over the base model performance.

---

*This project is part of the MLOps course curriculum at IITM BS Degree Program, focusing on practical machine learning operations and model evaluation techniques.*