import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('./data.csv', header=None)
dataset.describe()

for i in [24,72,168]:
	print "For %d-th hour: mean = %f, std = %f, median = %d" % (i,
																dataset[i].mean(),
																dataset[i].std(),
															 	dataset[i].median())


