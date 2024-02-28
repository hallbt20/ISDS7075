import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from math import sqrt


def average_last_values(df, x):
    # Group by 'engine_type'
    grouped = df.groupby('engine')

    # Initialize an empty DataFrame to store results
    result_df = pd.DataFrame(columns=['engine', 'rul', 'prob'])

    # Iterate over groups
    for name, group in grouped:
        # Calculate the average of the last x values for 'predicted_rul' and 'predicted_prob'
        tail = group.tail(x)
        row = {'engine': name, 'rul': round(tail['predicted_rul'].mean()), 'prob': tail['predicted_prob'].mean()}
        if result_df.empty:
            result_df = pd.DataFrame([row])
        else:
            result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

    return result_df


def penalty(x):
    if x < 0:
        return np.exp(-x / 10) - 1
    else:
        return np.exp(x / 13) - 1


""" Part 1: Data Preprocessing """
# Load the datasets
training_data = pd.read_csv('training_data.csv')
validation_data = pd.read_csv('validation_data.csv')
validation_rul = pd.read_csv('validation_rul_and_prod.csv')
test_data = pd.read_csv('test_data.csv')
test_rul = pd.read_csv('test_rul_and_prod.csv')

# Selecting sensor and operational setting columns for normalization
columns_to_normalize = ['s2', 's3', 'r1', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']
scaler = MinMaxScaler()
training_data[columns_to_normalize] = scaler.fit_transform(training_data[columns_to_normalize])
validation_data[columns_to_normalize] = scaler.transform(validation_data[columns_to_normalize])
test_data[columns_to_normalize] = scaler.transform(test_data[columns_to_normalize])

# Change r2 column to binary
training_data['r2'] = training_data['r2'].apply(lambda x: 0 if x == -1 else 1)
validation_data['r2'] = validation_data['r2'].apply(lambda x: 0 if x == -1 else 1)
test_data['r2'] = test_data['r2'].apply(lambda x: 0 if x == -1 else 1)

""" Part 2: Initializing data splitting for training, validation, and testing """
# Preparing the data for model training
pred_cols = ['day', 's1', 's2', 's3', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']
X_train = training_data[pred_cols]  # Features
X_val = validation_data[pred_cols]  # Features
X_test = test_data[pred_cols]  # Features

y_train_rul = training_data['rul']  # Target for regression
y_val_rul = validation_data['rul']  # Target for regression
y_test_rul = test_data['rul']  # Target for regression

y_train_prob = training_data['prob']  # Target for classification
y_val_prob = validation_data['prob']  # Target for classification
y_test_prob = test_data['prob']  # Target for classification

""" Part 3: Linear/Logistic Regression Models """
""" Step 1: Building the models """
# Initializing and training the Linear Regression model for RUL prediction
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_rul)
y_val_pred_rul = np.maximum(0, lr_model.predict(X_val))
y_test_pred_rul = np.maximum(0, lr_model.predict(X_test))

# Initializing and training the Logistic Regression model for critical condition classification
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train_prob)
y_val_pred_prob = log_reg_model.predict_proba(X_val)[:, 1]
y_test_pred_prob = log_reg_model.predict_proba(X_test)[:, 1]

""" Step 2: Determine performance measures for validation """
# Predicted values for validation
val_predictions = pd.DataFrame({
    'engine': validation_data['engine'],
    'day': validation_data['day'],
    'predicted_rul': y_val_pred_rul,
    'predicted_prob': y_val_pred_prob
})

for tail_value in range(1, 5):
    val_pred = average_last_values(val_predictions, tail_value)
    scaler = MinMaxScaler()
    val_pred['prob_adj'] = scaler.fit_transform(val_pred[['prob']])

    # Determine critical condition based on the maximum probability threshold
    critical_condition_threshold = 0.5
    val_pred['critical_condition'] = val_pred['prob_adj'] > critical_condition_threshold

    val_df = pd.DataFrame({'engine': validation_rul['engine']})

    val_df['rul_diff'] = validation_rul['rul'] - val_pred['rul']  # d_i values
    val_df['prob_diff_squared'] = pow(validation_rul['prob'] - val_pred['prob_adj'], 2)  # Summand for RMSE

    val_df['penalty'] = val_df['rul_diff'].apply(penalty)
    avg_penalty = val_df['penalty'].mean()

    RMSE = sqrt(val_df['prob_diff_squared'].mean())

    performance_measure = RMSE * avg_penalty

    print(f'When averaging the last {tail_value} values per engine, the performance measure is {performance_measure}.')

""" Step 3: Determine rul/prob values for test data """
# Predicted values for testing
test_predictions = pd.DataFrame({
    'engine': test_data['engine'],
    'day': test_data['day'],
    'predicted_rul': y_test_pred_rul,
    'predicted_prob': y_test_pred_prob
})

# For Test: Apply the functions to each group (engine)
test_pred = average_last_values(test_predictions, 4)
scaler = MinMaxScaler()
test_pred['prob_adj'] = scaler.fit_transform(test_pred[['prob']])

""" Part 3: Random Forest Models """
""" Step 1: Building the models """
# Initialize the Random Forest regressor for RUL prediction and train on training rul data
rf_regressor = RandomForestRegressor(
    n_estimators=2500,
    max_depth=20,
    random_state=42,
    min_samples_split=10,
    n_jobs=-1,
    bootstrap=True
)
rf_regressor.fit(X_train, y_train_rul)

# Predict RUL on validation and test data
y_val_pred_rul_rf = np.maximum(0, rf_regressor.predict(X_val))
y_test_pred_rul_rf = np.maximum(0, rf_regressor.predict(X_test))

# Initialize the Random Forest classifier for critical condition prediction and train on prob data
rf_classifier = RandomForestClassifier(
    n_estimators=2500,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    min_samples_split=10,
    bootstrap=True,
    class_weight='balanced'
)
rf_classifier.fit(X_train, y_train_prob)

# Predict probabilities of critical condition on validation and test data
y_val_pred_prob_rf = rf_classifier.predict_proba(X_val)[:, 1]
y_test_pred_prob_rf = rf_classifier.predict_proba(X_test)[:, 1]

""" Step 2: Determine performance measures for validation """
# Predicted values for validation
val_predictions = pd.DataFrame({
    'engine': validation_data['engine'],
    'day': validation_data['day'],
    'predicted_rul': y_val_pred_rul_rf,
    'predicted_prob': y_val_pred_prob_rf
})

for tail_value in range(1, 5):
    val_pred = average_last_values(val_predictions, tail_value)
    scaler = MinMaxScaler()
    val_pred['prob_adj'] = scaler.fit_transform(val_pred[['prob']])

    # Determine critical condition based on the maximum probability threshold
    critical_condition_threshold = 0.5
    val_pred['critical_condition'] = val_pred['prob_adj'] > critical_condition_threshold

    val_df = pd.DataFrame({'engine': validation_rul['engine']})
    val_df['rul_diff'] = validation_rul['rul'] - val_pred['rul']
    val_df['prob_diff_squared'] = pow(validation_rul['prob'] - val_pred['prob_adj'], 2)

    val_df['penalty'] = val_df['rul_diff'].apply(penalty)
    avg_penalty = val_df['penalty'].mean()

    RMSE = sqrt(val_df['prob_diff_squared'].mean())

    performance_measure = RMSE * avg_penalty

    print(f'When averaging the last {tail_value} values per engine, the performance measure is {performance_measure}.')

# Predicted values for testing
test_predictions = pd.DataFrame({
    'engine': test_data['engine'],
    'day': test_data['day'],
    'predicted_rul': y_test_pred_rul_rf,
    'predicted_prob': y_test_pred_prob_rf
})

# For Test: Get final predicted rul and prob values
test_pred = average_last_values(test_predictions, 3)
scaler = MinMaxScaler()
test_pred['prob_adj'] = scaler.fit_transform(test_pred[['prob']])


