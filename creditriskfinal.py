#!/usr/bin/env python
# coding: utf-8

# In[79]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
cr_loan = pd.read_csv('cr_loan2.csv')

# Check the structure of the data
print(cr_loan.dtypes)

# Check the first five rows of the data
cr_loan.head(5)


# In[80]:


# Looking at the distribution of loan amounts with a histogram
n, bins, patches = plt.hist(x=cr_loan['loan_amnt'], bins='auto', color='blue',alpha=0.7, rwidth=0.85)
plt.xlabel("Loan Amount")
plt.show()


# In[81]:


print("There are 32 000 rows of data so the scatter plot may take a little while to plot.")

# Plotting a scatter plot of income against age
plt.scatter(cr_loan['person_income'], cr_loan['person_age'],c='blue', alpha=0.5)
plt.xlabel('Personal Income')
plt.ylabel('Persone Age')
plt.show()


# In[82]:


# Creating a cross table of the loan intent and loan status
pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins = True)


# In[83]:


# Creating a cross table of home ownership, loan status, and grade
pd.crosstab(cr_loan['person_home_ownership'],[cr_loan['loan_status'],cr_loan['loan_grade']])


# In[84]:


# Create a cross table of home ownership, loan status, and average percent income
print(pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'],
              values=cr_loan['loan_percent_income'], aggfunc='mean'))


# In[85]:


# Create a box plot of percentage income by loan status
cr_loan.boxplot(column = ['loan_percent_income'], by = 'loan_status')
plt.title('Average Percent Income by Loan Status')
plt.suptitle('')
plt.show()


# In[86]:


# Create the cross table for loan status, home ownership, and the max employment length
print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],
        values=cr_loan['person_emp_length'], aggfunc='max'))


# In[87]:


# Create an array of indices where employment length is greater than 60
indices = cr_loan[cr_loan['person_emp_length'] > 60].index

# Drop the records from the data based on the indices and create a new dataframe
cr_loan_new = cr_loan.drop(indices)

# Create the cross table from earlier and include minimum employment length
print(pd.crosstab(cr_loan_new['loan_status'],cr_loan_new['person_home_ownership'],
            values=cr_loan_new['person_emp_length'], aggfunc=['min','max']))


# Generally with credit data, key columns like person_emp_length are of high quality, but there is always room for error. With this in mind, we build our intuition for detecting outliers!

# In[88]:


# Create the scatter plot for age and amount
plt.scatter(cr_loan['person_age'], cr_loan['loan_amnt'], c='blue', alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()


# In[89]:


import matplotlib

# Use Pandas to drop the record from the data frame and create a new one
cr_loan_new = cr_loan.drop(cr_loan[cr_loan['person_age'] > 80].index)

# Create a scatter plot of age and interest rate
colors = ["blue","red"]
plt.scatter(cr_loan_new['person_age'], cr_loan_new['loan_int_rate'],
            c = cr_loan_new['loan_status'],
            cmap = matplotlib.colors.ListedColormap(colors),
            alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Interest Rate")
plt.show()


# Notice that in the last plot we have loan_status as a label for colors. This shows a different color depending on the class. In this case, it's loan default and non-default, and it looks like there are more defaults with high interest

# In[90]:


# Print a null value column array
print(cr_loan.columns[cr_loan.isnull().any()])

# Print the top five rows with nulls for employment length
print(cr_loan[cr_loan['person_emp_length'].isnull()].head())

# # Impute the null values with the median value for all employment lengths
cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].median()), inplace=True)

# # Create a histogram of employment length
n, bins, patches = plt.hist(cr_loan['person_emp_length'], bins='auto', color='blue')
plt.xlabel("Person Employment Length")
plt.show()


# We can use several different functions like mean() and median() to replace missing data. The goal here is to keep as much of our data as we can! It's also important to check the distribution of that feature to see if it changed.

# In[91]:


# Print the number of nulls
print(cr_loan['loan_int_rate'].isnull().sum())

# Store the array on indices
indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index

# Save the new data without missing data
cr_loan_clean = cr_loan.drop(indices)


# In[92]:


cr_loan_clean.info()


# Now that the missing data and outliers have been processed, the data is ready for modeling! More often than not, financial data is fairly tidy, but it's always good to practice preparing data for analytical work.

# In[93]:


# Create two data sets for numeric and non-numeric data
cred_num = cr_loan_clean.select_dtypes(exclude=['object'])
cred_str = cr_loan_clean.select_dtypes(include=['object'])

# One-hot encode the non-numeric columns
cred_str_onehot = pd.get_dummies(cred_str)

# Union the one-hot encoded columns to the numeric ones
cr_loan_prep = pd.concat([cred_num, cred_str_onehot], axis=1)

# Print the columns in the new data set
print(cr_loan_prep.columns)


# In[94]:


cr_loan_prep.head()


# Trying Xgboost model

# In[95]:


X = cr_loan_prep.drop('loan_status', axis=1)
y = cr_loan_prep['loan_status']


# In[96]:


X.head()


# In[97]:


# Import the train_test_split function
from sklearn.model_selection import train_test_split

# Split the data into 40% test and 60% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[98]:


X_train.head()


# In[99]:


# Train a model
import xgboost as xgb
clf_gbt = xgb.XGBClassifier().fit(X_train, np.ravel(y_train))

# Predict with a model
gbt_preds = clf_gbt.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(gbt_preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))


# The predictions don't look the same as with the LogisticRegression(), do they? Notice that this model is already accurately predicting the probability of default for some loans with a true value of 1 in loan_status

# ![image.png](attachment:image.png)

# In[100]:


# Import the classification report method
from sklearn.metrics import classification_report

# Predict the labels for loan status
gbt_preds = clf_gbt.predict(X_test)

# Check the values created by the predict method
print(gbt_preds)

# Print the classification report of the model
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))


# Have a look at the precision and recall scores! Remember the low default recall values we were getting from the LogisticRegression()? This model already appears to have serious potential.

# In[101]:


X = cr_loan_prep[['person_income','loan_int_rate',
                  'loan_percent_income','loan_amnt',
                  'person_home_ownership_MORTGAGE','loan_grade_F']]


# In[102]:


y = cr_loan_prep[['loan_status']]


# In[103]:


# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)


# In[104]:


# Create and train the model on the training data
clf_gbt = xgb.XGBClassifier().fit(X_train,np.ravel(y_train))

# Print the column importances from the model
print(clf_gbt.get_booster().get_score(importance_type = 'weight'))


# So, the importance for loan_grade_F is only 23 in this case. This could be because there are so few of the F-grade loans. While the F-grade loans don't add much to predictions here, they might affect the importance of other training columns.

# In[105]:


X2 = cr_loan_prep[['loan_int_rate','person_emp_length']]
X3 = cr_loan_prep[['person_income','loan_int_rate','loan_percent_income']]


# In[106]:


# Create the training and testing sets X2` and `X3
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=.4, random_state=123)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=.4, random_state=123)


# In[107]:


# Train a model on the X data with 2 columns
clf_gbt2 = xgb.XGBClassifier().fit(X2_train,np.ravel(y_train))

# Plot the column importance for this model
xgb.plot_importance(clf_gbt2, importance_type = 'weight')
plt.show()


# In[108]:


# Train a model on the X data with 3 columns
clf_gbt3 = xgb.XGBClassifier().fit(X3_train,np.ravel(y_train))

# Plot the column importance for this model
xgb.plot_importance(clf_gbt3, importance_type = 'weight')
plt.show()


# The importance of loan_int_rate went down. Initially, this was the most important column, but person_income ended up taking the top spot here.

# In[109]:


# Predict the loan_status using each model
gbt_preds = clf_gbt.predict(X_test)
gbt2_preds = clf_gbt2.predict(X2_test)

# # Print the classification report of the first model
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))

# # Print the classification report of the second model
print(classification_report(y_test, gbt2_preds, target_names=target_names))


# The first model has an accuracy of 0.89, meaning it correctly predicted the loan status 89% of the time. For the 'Non-Default' class, it has a precision of 0.90 and recall of 0.96. For the 'Default' class, it has a precision of 0.82 and recall of 0.64. The F1-score, which is the harmonic mean of precision and recall, is 0.93 for 'Non-Default' and 0.72 for 'Default'.
# 
# The second model has an accuracy of 0.81. For the 'Non-Default' class, it has a precision of 0.82 and recall of 0.97. For the 'Default' class, it has a precision of 0.69 and recall of 0.26. The F1-score is 0.89 for 'Non-Default' and 0.38 for 'Default'.
# 
# In general, the first model performs better than the second one in terms of accuracy, precision, recall, and F1-score. However, the choice of model may depend on the specific requirements of your project. For example, if it's more important to correctly identify defaults (even if it means incorrectly classifying some non-defaults), you might prefer a model with a higher recall for the 'Default' class.

# We can now look at the crossvalidation to see if the score can be improved.

# In[110]:


params = {'objective': 'binary:logistic', 'seed': 123, 'eval_metric': 'auc'}

# Set the values for number of folds and stopping iterations
n_folds = 5
early_stopping = 10

# Create the DTrain matrix for XGBoost
DTrain = xgb.DMatrix(X_train, label = y_train)

# Create the data frame of cross validations
cv_df = xgb.cv(params, DTrain, num_boost_round = 5, nfold=n_folds,
            early_stopping_rounds=early_stopping)

# Print the cross validations data frame
print(cv_df)


# The AUC for both train-auc-mean and test-auc-mean improves at each iteration of cross-validation. As the iterations progress the scores get better, but will they eventually reach 1.0

# In[111]:


cv = xgb.cv(params, DTrain, num_boost_round = 600, nfold=10,
            shuffle = True)


# In[ ]:


#convert the cv results to a data frame cv_results_big
cv_results_big = pd.DataFrame(cv)


# In[ ]:


# Print the first five rows of the CV results data frame
print(cv_results_big.head())

# Calculate the mean of the test AUC scores
print(np.mean(cv_results_big['test-auc-mean']).round(2))

# Plot the test AUC scores for each iteration
plt.plot(cv_results_big['test-auc-mean'])
plt.title('Test AUC Score Over 600 Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Test AUC Score')
plt.show()


# Notice that the test AUC score never quite reaches 1.0 and begins to decrease slightly after 100 iterations. This is because this much cross-validation can actually cause the model to become overfit. So, there is a limit to how much cross-validation you should to.

# In[ ]:


from sklearn.model_selection import cross_val_score
# Create a gradient boosted tree model using two hyperparameters
gbt = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 7)

# Calculate the cross validation scores for 4 folds
cv_scores = cross_val_score(gbt, X_train, np.ravel(y_train), cv = 4)

# Print the cross validation scores
print(cv_scores)

# Print the average accuracy and standard deviation of the scores
print("Average accuracy: %0.2f (+/- %0.2f)" % (np.mean(cv_scores),
                                              np.std(cv_scores) * 2))


# average cv_score for this course is getting higher! With only a couple of hyperparameters and cross-validation, we can get the average accuracy up to 89%. This is a great way to validate how robust the model is.

# In machine learning, having an imbalanced dataset means that the classes in the target variable are not represented equally. For example, in a binary classification problem, you might have 95% of samples belonging to Class A and only 5% belonging to Class B. This imbalance can lead to biased models, as they tend to predict the majority class more often, resulting in poor performance when predicting the minority class.
# 
# Undersampling is one technique used to handle imbalanced datasets. It works by randomly removing samples from the majority class to balance the class distribution. In the provided code, nondefaults are being undersampled to match the number of defaults, which are presumably the minority class.
# 
# While undersampling can help improve the model's performance on the minority class, it's not without its drawbacks. The main one is that it can lead to loss of information, as it removes potentially useful data. It can also introduce bias if the samples removed contain unique characteristics not present in the remaining samples.
# 
# Therefore, it's important to use undersampling judiciously and consider other techniques like oversampling (adding more samples to the minority class), SMOTE (Synthetic Minority Over-sampling Technique), or using algorithms that are less sensitive to class imbalance.

# In[ ]:


X_y_train = pd.concat([X_train.reset_index(drop = True),
                       y_train.reset_index(drop = True)], axis = 1)
count_nondefault, count_default = X_y_train['loan_status'].value_counts()


# In[ ]:


# Create data sets for defaults and non-defaults
nondefaults = X_y_train[X_y_train['loan_status'] == 0]
defaults = X_y_train[X_y_train['loan_status'] == 1]

# Undersample the non-defaults
nondefaults_under = nondefaults.sample(count_default)

# Concatenate the undersampled nondefaults with defaults
X_y_train_under = pd.concat([nondefaults_under.reset_index(drop = True),
                             defaults.reset_index(drop = True)], axis = 0)

# Print the value counts for loan status
print(X_y_train_under['loan_status'].value_counts())


# Now, our training set has an even number of defaults and non-defaults. Let's test out some machine learning models on this new undersampled data set and compare their performance to the models trained on the regular data set.

# In[ ]:


#retrain the model with the undersampled data set by gb2_preds
X_train_under = X_y_train_under.drop('loan_status', axis = 1)
y_train_under = X_y_train_under['loan_status']


# In[ ]:


#split the data into training and testing sets
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_train_under, y_train_under, test_size=.4, random_state=123)


# In[ ]:


# Create a gradient boosted tree model with two hyperparameters
gbt2 = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 7).fit(X_train_under, np.ravel(y_train_under))

# Predict the loan_status using the model
gbt2_preds = gbt2.predict(X_test_under)


# In[ ]:


# Check the classification reports
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))
print(classification_report(y_test_under, gbt2_preds, target_names=target_names))


# In[ ]:


# Import the roc_auc_score method
from sklearn.metrics import roc_auc_score

# Print and compare the AUC scores of the old and new models
print(roc_auc_score(y_test, gbt_preds))
print(roc_auc_score(y_test_under, gbt2_preds))


# Undersampling the training data results in more false positives, but the recall for defaults and the AUC score are both higher than the original model. This means overall it predicts defaults much more accurately.

# In[ ]:


#get the predictions for the logistic regression and gradient boosted tree

clf_gbt_preds = gbt2.predict(X_test)


# In[ ]:


# ROC chart components
#import the roc_curve and auc methods
from sklearn.metrics import roc_curve, auc


fallout_gbt, sensitivity_gbt, thresholds_gbt = roc_curve(y_test, clf_gbt_preds)

# ROC Chart with both
plt.plot(fallout_gbt, sensitivity_gbt, color = 'green', label='%s' % 'GBT')
plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for LR and GBT on the Probability of Default")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()


# In[ ]:


# Create the calibration curve plot with the guideline
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()


# In[ ]:


#get the calibration curve for the logistic regression and gradient boosted tree
from sklearn.calibration import calibration_curve
mean_pred_val_gbt, frac_of_pos_gbt = calibration_curve(y_test, clf_gbt_preds, n_bins = 10)


# In[ ]:


# Add the calibration curve for the gradient boosted tree
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.plot(mean_pred_val_gbt, frac_of_pos_gbt,
         's-', label='%s' % 'Gradient Boosted tree')
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()


# In[ ]:


#save the gradient boosted tree model
import pickle
filename = 'finalized_model.sav'
pickle.dump(clf_gbt, open(filename, 'wb'))


# In[ ]:


#turn this notebook into a script
get_ipython().system('jupyter nbconvert --to script creditriskfinal.ipynb')

