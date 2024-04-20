#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


# Looking at the distribution of loan amounts with a histogram
n, bins, patches = plt.hist(x=cr_loan['loan_amnt'], bins='auto', color='blue',alpha=0.7, rwidth=0.85)
plt.xlabel("Loan Amount")
plt.show()


# In[4]:


print("There are 32 000 rows of data so the scatter plot may take a little while to plot.")

# Plotting a scatter plot of income against age
plt.scatter(cr_loan['person_income'], cr_loan['person_age'],c='blue', alpha=0.5)
plt.xlabel('Personal Income')
plt.ylabel('Persone Age')
plt.show()


# In[5]:


# Creating a cross table of the loan intent and loan status
pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins = True)


# In[6]:


# Creating a cross table of home ownership, loan status, and grade
pd.crosstab(cr_loan['person_home_ownership'],[cr_loan['loan_status'],cr_loan['loan_grade']])


# In[7]:


# Create a cross table of home ownership, loan status, and average percent income
print(pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'],
              values=cr_loan['loan_percent_income'], aggfunc='mean'))


# In[8]:


# Create a box plot of percentage income by loan status
cr_loan.boxplot(column = ['loan_percent_income'], by = 'loan_status')
plt.title('Average Percent Income by Loan Status')
plt.suptitle('')
plt.show()


# In[9]:


# Create the cross table for loan status, home ownership, and the max employment length
print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],
        values=cr_loan['person_emp_length'], aggfunc='max'))


# In[10]:


# Create an array of indices where employment length is greater than 60
indices = cr_loan[cr_loan['person_emp_length'] > 60].index

# Drop the records from the data based on the indices and create a new dataframe
cr_loan_new = cr_loan.drop(indices)

# Create the cross table from earlier and include minimum employment length
print(pd.crosstab(cr_loan_new['loan_status'],cr_loan_new['person_home_ownership'],
            values=cr_loan_new['person_emp_length'], aggfunc=['min','max']))


# Generally with credit data, key columns like person_emp_length are of high quality, but there is always room for error. With this in mind, we build our intuition for detecting outliers!

# In[11]:


# Create the scatter plot for age and amount
plt.scatter(cr_loan['person_age'], cr_loan['loan_amnt'], c='blue', alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()


# In[12]:


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

# In[13]:


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

# In[14]:


# Print the number of nulls
print(cr_loan['loan_int_rate'].isnull().sum())

# Store the array on indices
indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index

# Save the new data without missing data
cr_loan_clean = cr_loan.drop(indices)


# Now that the missing data and outliers have been processed, the data is ready for modeling! More often than not, financial data is fairly tidy, but it's always good to practice preparing data for analytical work.

# In[15]:


from sklearn.linear_model import LogisticRegression

# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate']]
y = cr_loan_clean[['loan_status']]

# Create and fit a logistic regression model
clf_logistic_single = LogisticRegression()
clf_logistic_single.fit(X, np.ravel(y))

# Print the parameters of the model
print(clf_logistic_single.get_params())

# Print the intercept of the model
print(clf_logistic_single.intercept_)


# In[16]:


# Create X data for the model
X_multi = cr_loan_clean[['loan_int_rate','person_emp_length']]

# Create a set of y data for training
y = cr_loan_clean[['loan_status']]

# Create and train a new logistic regression
clf_logistic_multi = LogisticRegression(solver='lbfgs').fit(X_multi, np.ravel(y))

# Print the intercept of the model
print(clf_logistic_multi.intercept_)


# Take a closer look at each model's .intercept_ value. The values have changed! The new clf_logistic_multi model has an .intercept_ value closer to zero. This means the log odds of a non-default is approaching zero.

# In[17]:


from sklearn.model_selection import train_test_split

# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate','person_emp_length','person_income']]
y = cr_loan_clean[['loan_status']]

# Use test_train_split to create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

# Create and fit the logistic regression model
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Print the models coefficients
print(clf_logistic.coef_)


# We see that three columns were used for training and there are three values in .coef_? This tells us how important each column, or feature, was for predicting. The more positive the value, the more it predicts defaults. Look at the value for loan_int_rate.

# Creating a onehot encoding for the categorical variables

# In[18]:


# Create two data sets for numeric and non-numeric data
cred_num = cr_loan_clean.select_dtypes(exclude=['object'])
cred_str = cr_loan_clean.select_dtypes(include=['object'])

# One-hot encode the non-numeric columns
cred_str_onehot = pd.get_dummies(cred_str)

# Union the one-hot encoded columns to the numeric ones
cr_loan_prep = pd.concat([cred_num, cred_str_onehot], axis=1)

# Print the columns in the new data set
print(cr_loan_prep.columns)


# In[19]:


# Train the logistic regression model on the training data
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Create predictions of probability for loan status using test data
preds = clf_logistic.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))


# We have some predictions now, but they don't look very accurate. It looks like most of the rows with loan_status at 1 have a low probability of default. How good are the rest of the predictions. Next, let's see if we can determine how accurate the entire model is.

# In[20]:


from sklearn.metrics import roc_curve, roc_auc_score , classification_report
# Create a dataframe for the probabilities of default
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])

# # Reassign loan status based on the threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

# # Print the row counts for each loan status
print(preds_df['loan_status'].value_counts())


# In[21]:


# # Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))


# In[22]:


from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
# Print all the non-average values from the report
print(precision_recall_fscore_support(y_test,preds_df['loan_status']))


# In[23]:


# Print the first two numbers from the report
print(precision_recall_fscore_support(y_test,preds_df['loan_status'])[0])


# Now we know how to pull out specific values from the report to either store later for comparison, or use to check against portfolio performance. We can look at the impact of recall for defaults. This way, we can store that value for later calculations.

# In[24]:


# Create predictions and store them in a variable
preds = clf_logistic.predict_proba(X_test)

# Print the accuracy score the model
print(clf_logistic.score(X_test, y_test))

# Plot the ROC curve of the probabilities of default
prob_default = preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()

# Compute the AUC and store it in a variable
auc = roc_auc_score(y_test, prob_default)


# In[ ]:





#  So the accuracy for this model is about 80% and the AUC score is 76%. Notice that what the ROC chart shows us is the tradeoff between all values of our false positive rate (fallout) and true positive rate (sensitivity).

# In[25]:


# Set the threshold for defaults to 0.5
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test,preds_df['loan_status']))


# In[26]:


# Set the threshold for defaults to 0.4
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test,preds_df['loan_status']))


# In[27]:


#calculate the average loan amount
avg_loan_amnt = cr_loan['loan_amnt'].mean()
avg_loan_amnt


# In[28]:


#create a new dataframe preds_def with the probability of default and loan status
preds_def = pd.DataFrame(preds[:,1], columns = ['prob_default'])
preds_def['loan_status'] = preds_def['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

preds_def.head()


# In[29]:


# Reassign the values of loan status based on the new threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Store the number of loan defaults from the prediction data
num_defaults = preds_df['loan_status'].value_counts()[1]

# Store the default recall from the classification report
default_recall = precision_recall_fscore_support(y_test,preds_df['loan_status'])[1][1]

# Calculate the estimated impact of the new default recall rate
print(avg_loan_amnt * num_defaults * (1 - default_recall))


# In[30]:


clf_logistic_preds = clf_logistic.predict(X_test)


# By our estimates, this loss would be around $12.2 million. Try rerunning this code with threshold values of 0.3 and 0.5. Do you see the estimated losses changing? How do we find a good threshold value based on these metrics alone?

# Trying Xgboost model

# In[31]:


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

# In[32]:


#create a new dataframe called portfolio with gbt_prob_default,lr_prob_default,lgd of 0.2,loan_amnt
portfolio = pd.DataFrame(gbt_preds[:,1], columns = ['gbt_prob_default'])
portfolio['lr_prob_default'] = preds[:,1]
portfolio['lgd'] = 0.2
portfolio['loan_amnt'] = cr_loan_clean['loan_amnt']

#print the first five rows of the portfolio
print(portfolio.head())


# In[33]:


# Print the first five rows of the portfolio data frame
print(portfolio.head())

# Create expected loss columns for each model using the formula
portfolio['gbt_expected_loss'] = portfolio['gbt_prob_default'] * portfolio['lgd'] * portfolio['loan_amnt']
portfolio['lr_expected_loss'] = portfolio['lr_prob_default'] * portfolio['lgd'] * portfolio['loan_amnt']

# Print the sum of the expected loss for lr
print('LR expected loss: ', np.sum(portfolio['lr_expected_loss']))

# Print the sum of the expected loss for gbt
print('GBT expected loss: ', np.sum(portfolio['gbt_expected_loss']))


# In[34]:


# Predict the labels for loan status
gbt_preds = clf_gbt.predict(X_test)

# Check the values created by the predict method
print(gbt_preds)

# Print the classification report of the model
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))


# Have a look at the precision and recall scores! Remember the low default recall values we were getting from the LogisticRegression()? This model already appears to have serious potential.

# In[35]:


X = cr_loan_prep[['person_income','loan_int_rate',
                  'loan_percent_income','loan_amnt',
                  'person_home_ownership_MORTGAGE','loan_grade_F']]


# In[36]:


y = cr_loan_prep[['loan_status']]


# In[37]:


# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)


# In[38]:


# Create and train the model on the training data
clf_gbt = xgb.XGBClassifier().fit(X_train,np.ravel(y_train))

# Print the column importances from the model
print(clf_gbt.get_booster().get_score(importance_type = 'weight'))


# So, the importance for loan_grade_F is only 23 in this case. This could be because there are so few of the F-grade loans. While the F-grade loans don't add much to predictions here, they might affect the importance of other training columns.

# In[39]:


X2 = cr_loan_prep[['loan_int_rate','person_emp_length']]
X3 = cr_loan_prep[['person_income','loan_int_rate','loan_percent_income']]


# In[40]:


# Create the training and testing sets X2` and `X3
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=.4, random_state=123)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=.4, random_state=123)


# In[41]:


# Train a model on the X data with 2 columns
clf_gbt2 = xgb.XGBClassifier().fit(X2_train,np.ravel(y_train))

# Plot the column importance for this model
xgb.plot_importance(clf_gbt2, importance_type = 'weight')
plt.show()


# In[42]:


# Train a model on the X data with 3 columns
clf_gbt3 = xgb.XGBClassifier().fit(X3_train,np.ravel(y_train))

# Plot the column importance for this model
xgb.plot_importance(clf_gbt3, importance_type = 'weight')
plt.show()


# The importance of loan_int_rate went down. Initially, this was the most important column, but person_income ended up taking the top spot here.

# In[43]:


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

# In[44]:


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

# In[45]:


cv = xgb.cv(params, DTrain, num_boost_round = 600, nfold=10,
            shuffle = True)


# In[46]:


#convert the cv results to a data frame cv_results_big
cv_results_big = pd.DataFrame(cv)


# In[47]:


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

# In[48]:


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

# In[49]:


X_y_train = pd.concat([X_train.reset_index(drop = True),
                       y_train.reset_index(drop = True)], axis = 1)
count_nondefault, count_default = X_y_train['loan_status'].value_counts()


# In[50]:


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

# In[51]:


#retrain the model with the undersampled data set by gb2_preds
X_train_under = X_y_train_under.drop('loan_status', axis = 1)
y_train_under = X_y_train_under['loan_status']


# In[52]:


#split the data into training and testing sets
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_train_under, y_train_under, test_size=.4, random_state=123)


# In[53]:


# Create a gradient boosted tree model with two hyperparameters
gbt2 = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 7).fit(X_train_under, np.ravel(y_train_under))

# Predict the loan_status using the model
gbt2_preds = gbt2.predict(X_test_under)


# In[54]:


# Check the classification reports
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))
print(classification_report(y_test_under, gbt2_preds, target_names=target_names))


# In[55]:


# Print and compare the AUC scores of the old and new models
print(roc_auc_score(y_test, gbt_preds))
print(roc_auc_score(y_test_under, gbt2_preds))


# Undersampling the training data results in more false positives, but the recall for defaults and the AUC score are both higher than the original model. This means overall it predicts defaults much more accurately.

# In[56]:


#create a dataframe for predictions preds_df_lr and preds_df_gbt which has prob_default  loan_status
preds_df_lr = pd.DataFrame(preds[:,1], columns = ['prob_default'])
preds_df_lr['loan_status'] = preds_df_lr['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

preds_df_gbt = pd.DataFrame(gbt_preds, columns = ['prob_default'])
preds_df_gbt['loan_status'] = preds_df_gbt['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)


# In[57]:


# Print the logistic regression classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df_lr['loan_status'], target_names=target_names))

# Print the gradient boosted tree classification report
print(classification_report(y_test, preds_df_gbt['loan_status'], target_names=target_names))

# Print the default F-1 scores for the logistic regression
print(precision_recall_fscore_support(y_test,preds_df_lr['loan_status'], average = 'macro')[2])

# Print the default F-1 scores for the gradient boosted tree
print(precision_recall_fscore_support(y_test,preds_df_gbt['loan_status'], average = 'macro')[2])


# There is a noticeable difference between these two models. Do you see that the scores from the classification_report() are all higher for the gradient boosted tree? This means the tree model is better in all of these aspects. Let's check the ROC curve.

# In[58]:


#get the predictions for the logistic regression and gradient boosted tree

clf_gbt_preds = clf_gbt.predict(X_test)


# In[59]:


# ROC chart components
fallout_lr, sensitivity_lr, thresholds_lr = roc_curve(y_test, clf_logistic_preds)
fallout_gbt, sensitivity_gbt, thresholds_gbt = roc_curve(y_test, clf_gbt_preds)

# ROC Chart with both
plt.plot(fallout_lr, sensitivity_lr, color = 'blue', label='%s' % 'Logistic Regression')
plt.plot(fallout_gbt, sensitivity_gbt, color = 'green', label='%s' % 'GBT')
plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for LR and GBT on the Probability of Default")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()


# In[60]:


# Print the logistic regression AUC with formatting
print("Logistic Regression AUC Score: %0.2f" % roc_auc_score(y_test, clf_logistic_preds))

# Print the gradient boosted tree AUC with formatting
print("Gradient Boosted Tree AUC Score: %0.2f" % roc_auc_score(y_test, clf_gbt_preds))


# the ROC curve for the gradient boosted tree. Not only is the lift much higher, the calculated AUC score is also quite a bit higher. It's beginning to look like the gradient boosted tree is best. Let's check the calibration to be sure.

# In[61]:


# Create the calibration curve plot with the guideline
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()


# In[62]:


#get the calibration curve for the logistic regression and gradient boosted tree
from sklearn.calibration import calibration_curve
mean_pred_val_lr, frac_of_pos_lr = calibration_curve(y_test, clf_logistic_preds, n_bins = 10)
mean_pred_val_gbt, frac_of_pos_gbt = calibration_curve(y_test, clf_gbt_preds, n_bins = 10)


# In[63]:


mean_pred_val_lr


# In[64]:


# Add the calibration curve for the gradient boosted tree
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.plot(mean_pred_val_lr, frac_of_pos_lr,
         's-', label='%s' % 'Logistic Regression')
plt.plot(mean_pred_val_gbt, frac_of_pos_gbt,
         's-', label='%s' % 'Gradient Boosted tree')
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()


# In[65]:


#save the gradient boosted tree model
import pickle
filename = 'finalized_model.sav'
pickle.dump(clf_gbt, open(filename, 'wb'))


# In[ ]:


#turn this notebook into a script
get_ipython().system('jupyter nbconvert --to script creditrisk.ipynb')

