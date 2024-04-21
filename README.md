# Credit Risk Analysis and Loan Default Prediction App
![](https://github.com/abh2050/creditriskcomodeling/blob/main/pic1.png)
![](https://github.com/abh2050/creditriskcomodeling/blob/main/pic2.png)
This project consists of two components: a Jupyter notebook for credit risk analysis and a Streamlit app for real-time loan default prediction. The Jupyter notebook explores a credit loan dataset, applies data preprocessing techniques, and develops predictive models using logistic regression and gradient boosted trees (XGBoost). The Streamlit app allows users to interact with the pre-trained model and predict the probability of loan default based on loan characteristics.

## Jupyter Notebook
### Data Description
The dataset used in this project is `cr_loan2.csv`, which contains information about loans and personal characteristics. Key features include:
- `loan_amnt`: The amount of the loan.
- `loan_int_rate`: The loan's interest rate.
- `person_income`: The borrower's annual income.
- `person_emp_length`: The borrower's employment length in months.
- `loan_status`: Whether the loan is in default (1) or not (0).

### Data Preprocessing
The following preprocessing steps are applied in the Jupyter notebook:
- **Outlier Removal**: Rows with employment length greater than 60 months and ages over 80 years are removed.
- **Missing Data Handling**: Missing values in `person_emp_length` are imputed with the median. Rows with missing `loan_int_rate` are dropped.
- **Feature Engineering**: One-hot encoding is used for categorical variables, and a new feature `loan_percent_income` is derived as the ratio of loan amount to income.

### Exploratory Data Analysis (EDA)
The Jupyter notebook performs various exploratory analyses to understand the data structure:
- **Histograms**: For loan amounts and employment lengths.
- **Scatter Plots**: To explore relationships between personal income, age, and loan amounts.
- **Cross Tables**: To examine relationships between loan intent, loan status, home ownership, and loan grade.

### Predictive Models
The following models are developed and compared in the Jupyter notebook:
- **Logistic Regression**: A linear model for binary classification.
- **Gradient Boosted Trees (XGBoost)**: A more complex ensemble model.

### Model Evaluation
The models are evaluated using various metrics and visualizations:
- **Classification Reports**: Providing precision, recall, F1-score, and support for each class.
- **ROC Curves**: Plotting the true positive rate against the false positive rate to visualize model performance.
- **AUC Scores**: Quantifying the area under the ROC curve.
- **Calibration Curves**: Assessing the accuracy of predicted probabilities.
- **Confusion Matrix**: To visualize true positives, false positives, true negatives, and false negatives.

### Key Findings
- **Model Comparison**: The gradient boosted trees model consistently outperforms the logistic regression model in terms of accuracy, precision, recall, and F1-score.
- **Handling Imbalanced Data**: The notebook explores undersampling to balance the dataset, which improves the recall for defaults but may introduce more false positives.

### Output
The final model, a gradient boosted tree, is saved to `finalized_model.pkl`. This file is used in the Streamlit app for real-time loan default predictions.

## Streamlit App
### Functionality
The Streamlit app predicts the probability of loan default based on user inputs. It uses the pre-trained gradient boosted tree model saved from the Jupyter notebook.

### How to Run
1. **Launch Streamlit**: Open a terminal and navigate to the directory containing the Streamlit script.
   - Run the following command to start the app: `streamlit run app.py`
   - Ensure that `finalized_model.pkl` is in the same directory as the Streamlit script.
2. **Input Loan Details**:
   - Provide values for loan amount, loan interest rate, and personal income.
   - Indicate whether there's a mortgage on the home and if the loan grade is F.
3. **Get Prediction**:
   - The app will display the predicted probability of loan default.
   - If the probability is above 50%, a warning will indicate a likely default; otherwise, a success message suggests a low risk of default.

### Dependencies
Ensure the following are installed to run the Streamlit app:
- Python 3.x
- Streamlit
- Pandas
- Numpy
- XGBoost
- Pickle

## Conclusion
This project demonstrates a comprehensive credit risk analysis and provides a real-time loan default prediction app. The gradient boosted trees model offers better predictions compared to logistic regression. Proper handling of imbalanced datasets and outlier removal are crucial for accurate model performance.

## Notes
- Further extensions could include more advanced feature engineering and hyperparameter tuning.
- The Streamlit app can be integrated into larger applications or dashboards.
- Data privacy and security should be a priority, especially with sensitive financial information.
