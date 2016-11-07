import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# functions' definitions
def m_rse(y_pred, y_true):
    return np.mean((y_pred / y_true -1) ** 2)

### STEP 1 ###
# reading the data and showing the summary
dataset = pd.read_csv('./data.csv', header=None)
dataset.describe()

# showing basic statistics for n=24, 72, 168
for i in [24,72,168]:
	print "For %d-th hour: mean = %f, std = %f, median = %d" % (i, dataset[i].mean(), dataset[i].std(), dataset[i].median())

### STEP 2 ###
# plotting the distribution of v(168)
dataset[168].hist(bins = 40)

### STEP 3 ###
# applying log_tranform
log_168 = dataset[168].apply(lambda x: np.log(x))
# plotting the log-transformed distribution of v(168)
log_168.hist(bins = 40)

### STEP 4 ###
# removing outliers
std_168 = log_168.std()
mean_168 = log_168.mean()

# selecting non-outliers
selector = np.abs(log_168 - mean_168) < 3 * std_168
dataset = dataset.loc[selector]

### STEP 5 ###
# computing correlation coefficients between the log-transformed v(n) for n=1,2,..,24 and v(168)
range_1_24 = range(1,25,1)
correlations = dataset[range_1_24].apply(lambda x: np.log(x+1)).apply(lambda x: log_168.corr(x))

# plotting the correlations
plt.plot(correlations, label='Correlation')
plt.xlabel('n')
plt.ylabel('Correlation value')
plt.title('Dependency between n and correlation')

### STEP 6 ###
# applying log-tranform
range_for_log_transform = range(1, dataset.shape[1])
dataset[range_for_log_transform] = dataset[range_for_log_transform].apply(lambda x: np.log(x+1))

# train-test split procedure
train_test_permutation = np.random.permutation(dataset.shape[0])
split_index = int(0.1 * dataset.shape[0])

test_indices = train_test_permutation[:split_index]
train_indices = train_test_permutation[split_index:]

# dividing dataset into train and test
test, train = dataset.iloc[test_indices], dataset.iloc[train_indices]

### STEP 7 ###
# Using log-transformed training data, find linear regression model that minimizes Ordinary
# Least Squares (OLS) error function.
list_of_models = [LinearRegression() for x in xrange(24)]

#training
for i in xrange(24):
    list_of_models[i].fit(train[i+1].reshape(-1,1), train[168])

#Error computation
linear_input_error = [0] * 24

for i in xrange(24):
    linear_input_error[i] = m_rse(list_of_models[i].predict(test[i + 1].reshape(-1, 1)), test[168])

### STEP 8 ###
# Extending the linear regression model with multiple inputs.
list_of_models_multiple_inputs = [LinearRegression() for x in xrange(24)]

#training
for i in xrange(24):
    current_range = range(1, i+2)
    list_of_models_multiple_inputs[i].fit(train[current_range].as_matrix().reshape(-1,i+1), train[168])

### STEP 9 ###
# Computing the mRSE error
multiple_input_error = [0] * 24

for i in xrange(24):
    current_range = range(1, i+2)
    multiple_input_error[i] = m_rse(list_of_models_multiple_inputs[i].predict
                                    (test[current_range].as_matrix().reshape(-1, i + 1)),
                                    test[168])

### STEP 10 ###
# Plotting the mRSE values
plt.plot(range(1, 25), linear_input_error, label='Linear Regression', linestyle='solid', color='red', marker='o', markersize=4.0, linewidth=2.5)
plt.plot(range(1, 25), multiple_input_error, label='Multiple-input Linear Regression', linestyle='dashed', marker='x', markersize=4.0, color='blue', linewidth=2.5)
plt.legend()
plt.xlabel('Reference time (n)', fontsize=12)
plt.ylabel('mRSE', fontsize=12)

fig_size = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = (18,12)

