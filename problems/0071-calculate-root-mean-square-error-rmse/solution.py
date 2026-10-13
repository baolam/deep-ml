
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	rmse_res = np.sqrt(((y_true - y_pred) ** 2).mean())
	rmse_res = float(rmse_res)
	return round(rmse_res,3)
