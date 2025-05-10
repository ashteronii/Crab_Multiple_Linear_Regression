# Crab Age Prediction using Multiple Linear Regression

This project applies both a custom implementation of Ordinary Least Squares (OLS) Linear Regression and Scikit-learn's regression model to predict the age of crabs based on selected physical features. The aim is to explore how well linear models can estimate crab age using measurable attributes like shell weight and body dimensions.

---

## Dataset

- Gursewak Singh Sidhu (2022). *Crab Age Prediction*. Kaggle. https://www.kaggle.com/datasets/sidhus/crab-age-prediction/data
- The dataset contains physical characteristics of crabs.
- Target variable: `Age` (measured in months)
- Input features: Measurable physical characteristics of the crabs.

---

## Feature Selection

To determine which features to include in the model, a correlation matrix between the available attributes and the target variable (Age) was computed. Based on this analysis, we selected:
- `Shell Weight`
- `Diameter`
- `Height`

These features showed the strongest positive correlation with crab age while minimizing multicollinearity with each other. These attributes are also logically relevant, as crab size and weight tend to increase with age.

Predicting crab age from physical traits can help crab farmers monitor growth and make informed decisions about harvest timing.

---

## Project Structure

- `CrabAgePrediction.csv`: Raw dataset from Kaggle.
- `Crab_Cleaned.csv`: Cleaned dataset used for training.
- `Crab_Cleaned.py`: Data cleaning script.
- `Crab_Multiple_Linear_Regression.py`: Model training and evaluation script.
- `README.md`: This file.
- `Linear_Regression.py`: Custom OLS linear regression class implementation.

---

## Model Overview

Two models were applied:
1. Scikit-learn Linear Regression:
   - Uses the built-in LinearRegression model from sklearn.linear_model.
2. Custom Linear Regression:
   - Manually implemented using NumPy and the Normal Equation.
Both models were trained on the same feature matrix and evaluated using R^2 and RMSE.

---

## Results

| Model              | R^2 Score | RMSE   |
| ------------------ | -------- | ------ |
| Scikit-learn Model | \~0.3968 | \~2.50 |
| Custom OLS Model   | \~0.3968 | \~2.50 |

- R^2 Score (~39.7%): Indicates that the model explains only about 40% of the variance in crab age. This suggests that while there is some predictive power, other factors likely contribute to age that are not captured by the selected features.
- RMSE (~2.5 months): On average, predictions are off by ±2.5 months. This error might be acceptable depending on the harvesting cycle or precision requirements of crab farming operations.

---

## How to Run

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/ashteronii/Crab_Multiple_Linear_Regression.git
   cd Crab_Multiple_Linear_Regression
   ```
2. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Run the cleaning script:
   ```bash
   python Crab_Cleaned.py
   ```
4. Run the main script:
   ```bash
   python Crab_Multiple_Linear_Regression.py
   ```
   
---

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn

