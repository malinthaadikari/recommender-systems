import numpy as np
from LapLRS import LapLRSFunc
from Evaluating import evaluate
from DataPreProcess import data_pre_processor

# Obtain training set and the test set
pre_processed_data = data_pre_processor('resources/data/ml-100k/u.data', 1)

# obtain W
weight_matrix = LapLRSFunc(np.asmatrix(pre_processed_data[0]))

# obtain HR and ARHR
HR, ARHR = evaluate(weight_matrix, np.asmatrix(pre_processed_data[0]),  pre_processed_data[1])

print("HR is :%s",HR)
print("ARHR is :%s",ARHR)
