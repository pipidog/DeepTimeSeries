'''
this is an example file to show how to use DeepTimeSeries. The dataset was download from:
http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

This is a single household power comsumption data. The data has been cleaned and resampled.
The dataset used here is "clean_data.csv" 
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from DeepTimeSeries.models import RNN2Dense, Seq2Seq_1, Seq2Seq_2
from DeepTimeSeries.utils import series_to_superviesed, load_time_series_model
import numpy as np
import matplotlib.pyplot as plt

# hyperparameters =======================
task = 'train'   # 'train' / 'predict'
n_memory_steps = 10
n_forcast_steps = 10
train_split = 0.8
batch_size = 72
epochs = 10
test_model = 'Seq2Seq_1'  # 'RNN2Dense' / 'Seq2Seq'

# =======================================
# load data 
df = pd.read_csv('clean_data.csv', index_col=0, header=0)
values = df.values  

# scale data between 0 ~ 1 for better training results
data_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = data_scaler.fit_transform(values)

x_timeseries = scaled_values
y_timeseries = scaled_values[:,1].reshape(-1,1)

# convert to supervised learning compatible format
x_train, y_train, x_test, y_test = \
	series_to_superviesed(x_timeseries, y_timeseries, n_memory_steps, n_forcast_steps, split = 0.8)
print('size of x_train, y_train, x_test, y_test')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape )

# train model & inference
if task == 'train':
	# build model
	if test_model == 'RNN2Dense':
		model = RNN2Dense(x_train.shape[1:], y_train.shape[1:], 'GRU', 300, (20,))
	elif test_model == 'Seq2Seq_1':
		model = Seq2Seq_1(x_train.shape[1:], y_train.shape[1:], 'GRU', 300)
	elif test_model == 'Seq2Seq_2':
		model = Seq2Seq_2(x_train.shape[1:], y_train.shape[1:], 'GRU', 300)
	print(model.summary())

	# compile model
	model.compile()
	# train model
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
	   verbose=2, validation_data=(x_test,y_test))
	# save model
	model.save(test_model)
elif task == 'predict':
	# reload model 
	model = load_time_series_model(test_model)
	# predict data
	y_pred = model.predict(x_test)
	# plot results
	for n in range(n_forcast_steps):
		plt.subplot(n_forcast_steps,1,n+1)
		plt.plot(y_test[500:800,n,:],'b', label = 'True')
		plt.plot(y_pred[500:800,n,:],'r', label = 'Predict')
		plt.legend()
	plt.show()