import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def m_rse(y_pred, y_true):
    return np.mean((y_pred / y_true -1) ** 2)

dataset = pd.read_csv('./data.csv', header=None)
dataset.describe()

for i in [24,72,168]:
	print "For %d-th hour: mean = %f, std = %f, median = %d" % (i,
																dataset[i].mean(),
																dataset[i].std(),
															 	dataset[i].median())

dataset[168].hist(bins = 40)

log_168 = dataset[168].apply(lambda x: np.log(x))
log_168.hist(bins = 40)


std_168 = log_168.std()
mean_168 = log_168.mean()

selector = np.abs(log_168 - mean_168) < 3 * std_168
dataset = dataset.loc[selector]

range_1_24 = range(1,25,1)

correlations = dataset[range_1_24].apply(lambda x: np.log(x+1)).apply(lambda x: log_168.corr(x))
correlations

plt.plot(correlations, label='Correlation')
plt.xlabel('n')
plt.ylabel('Correlation value')
plt.title('Dependency between n and correlation')

range_for_log_transform = range(1, dataset.shape[1])
dataset[range_for_log_transform] = dataset[range_for_log_transform].apply(lambda x: np.log(x+1))

train_test_permutation = np.random.permutation(dataset.shape[0])
split_index = int(0.1 * dataset.shape[0])

test_indices = train_test_permutation[:split_index]
train_indices = train_test_permutation[split_index:]

test, train = dataset.iloc[test_indices], dataset.iloc[train_indices]

list_of_models = [LinearRegression() for x in xrange(24)]

for i in xrange(24):
    list_of_models[i].fit(train[i+1].reshape(-1,1), train[168])

linear_input_error = [0] * 24

for i in xrange(24):
    linear_input_error[i] = m_rse(list_of_models[i].predict(test[i + 1].reshape(-1, 1)), test[168])

list_of_models_multiple_inputs = [LinearRegression() for x in xrange(24)]

for i in xrange(24):
    current_range = range(1, i+2)
    list_of_models_multiple_inputs[i].fit(train[current_range].as_matrix().reshape(-1,i+1), train[168])

multiple_input_error = [0] * 24

for i in xrange(24):
    current_range = range(1, i+2)
    multiple_input_error[i] = m_rse(list_of_models_multiple_inputs[i].predict
                                    (test[current_range].as_matrix().reshape(-1, i + 1)),
                                    test[168])

plt.plot(range(1, 25), linear_input_error, label='Linear Regression', linestyle='solid', color='red', marker='o', markersize=4.0, linewidth=2.5)
plt.plot(range(1, 25), multiple_input_error, label='Multiple-input Linear Regression', linestyle='dashed', marker='x', markersize=4.0, color='blue', linewidth=2.5)
plt.legend()
plt.xlabel('Reference time (n)', fontsize=12)
plt.ylabel('mRSE', fontsize=12)

fig_size = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = (18,12)

