## Data Preprocessing 

### 1. Understanding the Basics
- Why is data preprocessing essential?
- Types of data:
  - Numerical
  - Categorical
  - Textual
  - Datetime

### 2. Data Loading and Initial Inspection
- Load data from CSV, Excel, JSON, etc.
- Inspect with `.head()`, `.info()`, `.describe()`
- Identify missing values, data types, and potential issues

### 3. Handling Missing Data
- Deletion: row-wise or column-wise
- Imputation:
  - Mean, median, mode
  - Forward/Backward fill
  - Advanced: `KNNImputer`, `IterativeImputer` from scikit-learn

### 4. Data Cleaning
- Remove duplicates
- Fix inconsistent entries
- Trim whitespaces, fix encoding issues
- Format datetime values

### 5. Data Transformation
- Log, square root, or other numerical transformations
- Encoding categorical features:
  - Label Encoding
  - One-Hot Encoding
  - Ordinal Encoding
- Feature Scaling:
  - `StandardScaler`
  - `MinMaxScaler`
  - `RobustScaler`, etc.

### 6. Feature Engineering
- Creating new features from existing ones
- Extracting features from datetime (day, month, weekday, etc.)
- Binning and polynomial features

### 7. Outlier Detection and Treatment
- IQR method
- Z-score method
- Visual analysis (boxplot, scatterplot)

### 8. Text Data Preprocessing (for NLP)
- Lowercasing, removing punctuation
- Removing stopwords
- Tokenization
- Stemming vs Lemmatization

### 9. Time Series Data Preprocessing
- Parsing and indexing datetime
- Rolling means and smoothing
- Seasonal decomposition

### 10. Train-Test Split and Data Partitioning
- Use `train_test_split` from scikit-learn
- Stratified sampling for imbalanced datasets
- Prepare for cross-validation

---

## Recommended Libraries
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (especially `sklearn.preprocessing`, `sklearn.impute`)
- `nltk`, `spaCy` (for text data)
- `statsmodels` (for time series)

---

## Suggested Learning Resources
- **Book**: *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron  
- **Course**: [Data Preprocessing for Machine Learning in Python - Coursera](https://www.coursera.org/learn/data-preprocessing-machine-learning)  
- **GitHub Repos**: `feature-engine`, `pandas-profiling`, `scikit-learn` examples
