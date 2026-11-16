# Bike Rental Demand Predictor 🚲

An exploratory machine learning project focused on predicting hourly bike rental demand using historical data. The project explores regression modelling, time-aware validation, hyperparameter tuning and error analysis to investigate how accurately bike rental demand can be predicted.

## Project Overview

The objective of this project was to develop a regression model capable of predicting the total number of bike rentals (`cnt`) based on historical information, including:

- Time-related features (hour, month, season, weekday).
- Calendar information (working days and holidays).
- Weather conditions (temperature, humidity, wind speed and weather situation).

The project follows a complete machine learning workflow, from initial data exploration to model evaluation and feature engineering.

## Dataset

The project uses the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), accessed through the `ucimlrepo` Python package.

The hourly dataset contains 17,379 observations covering bike rental activity during 2011 and 2012.

## Methodology

The project was developed through the following stages:

1. **Exploratory Data Analysis:** Investigated feature distributions, correlations and hourly, seasonal and weather-related rental patterns.
2. **Baseline Modelling:** Established a median-prediction baseline and evaluated an initial Multiple Linear Regression model.
3. **Random Forest Regression:** Developed a Random Forest Regressor to capture nonlinear relationships and interactions between features.
4. **Walk-Forward Validation:** Implemented chronological validation to evaluate model performance when predicting future observations using historical training data.
5. **Hyperparameter Tuning:** Experimented with different combinations of estimators, tree depth, minimum samples per split and minimum samples per leaf.
6. **Error Analysis:** Investigated residuals and prediction errors across different hours, working days and commuting periods.
7. **Feature Engineering:** Introduced an additional binary feature to identify working-day commuting hours and evaluated its effect on prediction performance.

## Model Performance

The models were initially compared using a chronological 80/20 train/test split.

| Model | MAE | R² |
|---|---:|---:|
| Median Baseline | 182.88 | -0.2901 |
| Multiple Linear Regression | 138.29 | 0.3090 |
| Random Forest Regressor | **44.99** | **0.9005** |

The Random Forest Regressor substantially outperformed both the baseline and the initial linear regression model.

### Walk-Forward Validation

The original Random Forest achieved the following average performance across the three validation folds:

- **Average MAE:** 53.90
- **Average R²:** 0.7877

When all validation predictions were combined, the model achieved an MAE of 58.52 and an R² of 0.7315.

Several hyperparameter configurations were tested, but none produced a meaningful and consistent improvement over the original model.

A lighter alternative using 50 estimators and a maximum depth of 25 achieved very similar predictive performance, while containing approximately 46.9% fewer tree nodes.

## Key Findings

- Random Forest performed considerably better than the initial baseline and linear regression models.
- The original configuration, using 100 estimators and the default values for the other tested hyperparameters, remained the strongest overall candidate.
- Prediction errors were particularly high during working-day commuting hours, especially around 8 AM and 5–6 PM.
- Although commuting periods represented approximately 8.5% of validation observations, they contributed around 23% of the total absolute prediction error.
- The model showed a tendency to underestimate rental demand during these peak periods.
- Introducing an `is_commute_period` feature changed the model's predictions and improved some commuting-hour error metrics, but did not produce a meaningful improvement in overall walk-forward performance.
- Removing either `temp` or `atemp` individually did not produce a measurable change in the tested walk-forward performance.

## Limitations and Future Improvements

Despite achieving strong predictive performance on the holdout dataset, the model struggles to capture the magnitude of demand during certain peak periods.

The available features may not fully capture external influences such as local events, transport disruptions or other unexpected changes in demand.

Further improvements could investigate additional external information, alternative regression algorithms and more recent bike-sharing data to evaluate how well the model generalises to different conditions.

## Technologies Used

- **Python**
- **Pandas & NumPy** – Data manipulation and numerical operations
- **Matplotlib & Seaborn** – Data visualisation
- **Scikit-learn** – Regression models, evaluation metrics and validation
- **Jupyter Notebook** – Experimentation and documentation
- **UCI Machine Learning Repository** – Dataset acquisition

## Running the Project

Clone the repository and install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo jupyter
```

Launch Jupyter Notebook or JupyterLab and open the project notebook.

The dataset is retrieved directly from the UCI repository using `ucimlrepo`, so an internet connection is required when loading the data.

---

*This project served as a practical exploration of regression modelling, time-aware model evaluation and evidence-based experimentation in machine learning.*
