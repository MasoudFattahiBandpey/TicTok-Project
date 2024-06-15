# TikTok Video Classification Project

This project aims to build a machine learning model to determine whether a TikTok video contains a claim or offers an opinion. This model helps reduce the backlog of user reports by prioritizing them more efficiently.

## Project Overview

TikTok users can report videos that they believe violate the platform's terms of service. To assist human moderators, this project utilizes machine learning techniques to classify videos as either claims or opinions. This classification helps streamline the moderation process by focusing on videos that are more likely to require review.

## Key Components

### 1. Ethical Considerations
- Address ethical implications of the model.
- Evaluate the consequences of false positives and false negatives.

### 2. Feature Engineering
- Select, extract, and transform features to prepare data for modeling.
- Extract text features from video transcriptions using CountVectorizer.

### 3. Modeling
- Build and evaluate machine learning models, including Random Forest and XGBoost.
- Select the best model based on recall and other performance metrics.

## Installation

To run this project, you will need the following libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
## Usage

1. **Load the Data**: Load the TikTok dataset into a Pandas DataFrame.
2. **Preprocess the Data**: Handle missing values, encode categorical variables, and perform feature engineering.
3. **Split the Data**: Split the data into training, validation, and test sets.
4. **Build Models**: Train Random Forest and XGBoost models, and select the best model based on performance metrics.
5. **Evaluate Models**: Evaluate the models using classification reports and confusion matrices.
```
## Results

- The Random Forest model achieved a high recall score, making it the preferred model.
- The model's most predictive features were related to user engagement metrics such as views, likes, shares, and downloads.

## Conclusion

This model effectively classifies TikTok videos as claims or opinions, aiding in the moderation process. Future improvements could include additional features like the number of times a video was reported.
