# Gold_prediction


# Gold Price Prediction using Random Forest

This project predicts gold prices using a machine learning model trained on historical market data. The Random Forest Regressor algorithm is used to achieve a high degree of accuracy in forecasting future gold prices based on features like stock market indices, oil prices, silver prices, and currency exchange rates.

## Project Overview

This project aims to:
- Analyze the correlation between gold prices and other market variables such as the S&P 500 Index (SPX), United States Oil Fund (USO), Silver ETF (SLV), and the EUR/USD exchange rate.
- Build a machine learning model to predict gold prices using historical data from 2008 to 2018.
- Evaluate the model's performance using metrics like R-squared (R²) and visualize the predicted versus actual gold prices.

## Dataset

The dataset used in this project is `gld_price_data.csv`, which contains the following columns:
- **Date**: Date of the record.
- **SPX**: S&P 500 Index.
- **GLD**: Gold price.
- **USO**: United States Oil Fund price.
- **SLV**: Silver ETF price.
- **EUR/USD**: Euro to US Dollar exchange rate.

The dataset includes 2290 rows, with each row representing a day’s data from 2008 to 2018.

## Dependencies

The project uses the following libraries:
- `pandas` for data manipulation and analysis.
- `numpy` for numerical operations.
- `scikit-learn` for machine learning model training and evaluation.
- `seaborn` and `matplotlib` for data visualization.

You can install all dependencies using the following command:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Model Development

The following steps were performed to build the model:

1. **Data Preprocessing**: 
   - The `Date` column is label-encoded to convert it into a numerical format.
   - No missing values were found in the dataset.

2. **Exploratory Data Analysis (EDA)**:
   - Correlation analysis showed a strong positive correlation between gold prices (`GLD`) and silver prices (`SLV`), and a weak negative correlation with oil prices (`USO`).

3. **Splitting Data**:
   - The data was split into training (80%) and testing (20%) sets using `train_test_split`.

4. **Model Training**:
   - A `RandomForestRegressor` model was trained on the training set.
   - The model achieved an R² score of **0.9988** on the training data and **0.9941** on the test data, indicating high accuracy.

5. **Model Evaluation**:
   - The model's predictions were compared with actual gold prices using R² score and visualized using a plot.

## Visualizations

- **Correlation Heatmap**: A heatmap of the correlations between different market variables.
- **Distribution Plot**: Shows the distribution of gold prices.
- **Actual vs Predicted Plot**: Compares the actual and predicted gold prices on the test set.

## Results

- The Random Forest model performed exceptionally well, achieving a near-perfect R² score on both the training and test datasets.
- The model successfully predicts gold prices with a high degree of accuracy, making it a useful tool for forecasting future gold prices.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Ashwadhama2004/Gold-Price-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Gold-Price-Prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook to execute the code and generate predictions:
   ```bash
   jupyter notebook
   ```

## Conclusion

This project demonstrates the power of Random Forest in predicting gold prices based on market data. The high accuracy achieved shows that the model captures the relationships between the features effectively.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
