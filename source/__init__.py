import numpy as np
from LapLRS import LapLRSFunc
from DataPreProcess import data_pre_processor

pre_processed_data = data_pre_processor('resources/data/ml-100k/u.data',1)
weight_matrix = LapLRSFunc(np.asmatrix(pre_processed_data[0]), pre_processed_data[1])