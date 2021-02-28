# first neural network with keras make predictions
from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import copy
from keras import backend as K
from math import sin, cos

# 0.87
def custom_activation1(x):
    return K.sin(x)

# 0.86
def custom_activation2(x):
    return K.sigmoid(sin(x) + cos(x))

# 0.88
def custom_activation3(x):
    return K.cos(K.sin(x) + 1)

# 0.81
def custom_activation4(x):
    return K.sin(x) * K.sin(x) * K.cos(x)

# 0.83
def custom_activation5(x):
    return K.sin(x) * K.cos(x) * K.cos(x)

# 0.88
def custom_activation6(x):
    return sin(x) - cos(x)

# 0.85
def custom_activation7(x):
    return K.sigmoid(sin(x)) * K.sigmoid(K.cos(x))

# 0.81
def custom_activation8(x):
    return K.sin(x) - K.sigmoid(K.sin(x) * K.cos(x))

# 0.85
def custom_activation9(x):
    return K.sin(x) + K.atan(1./x)

# 0.91
def custom_activation10(x):
    return K.atan(1./x)

# 0.87
def custom_activation11(x):
    return K.atan(x)

# 0.88
def custom_activation12(x):
    return K.atan(K.sin(x))

# 0.86
def custom_activation13(x):
    return K.atan(K.sin(x) + K.cos(x))

# 0.89
def custom_activation14(x):
    if x > 6.28:
        x % 6.28
    elif x < -1:
        -x / 10 + 6.28
    elif x < 0:
        x + 6.28
    else:
        x

# 0.85
def custom_activation15(x):
    if x > 6.28:
        x % 6.28
    elif x < -1:
        -x / 10 + 6.28
    elif x < 0:
        x + 6.28
    else:
        K.sin(x)

# 0.87
def custom_activation16(x):
    if x > 6.28:
        x % 6.28
    elif x < -1:
        -x / 10 + 6.28
    elif x < 0:
        x + 6.28
    else:
        K.sin(x) * K.cos(x)

# 0.86
def custom_activation17(x):
    if x > 6.28:
        x % 6.28
    elif x < -1:
        -x / 10 + 6.28
    elif x < 0:
        x + 6.28
    else:
        K.sin(x) * K.sin(x) * K.cos(x)

input_dim = 44

dataset = loadtxt('points30_1000_sincos_poly.csv', delimiter=',')
x = dataset[:,0:input_dim]
y = dataset[:,input_dim]

model = Sequential()
model.add(Dense(4, input_dim=input_dim, activation='elu'))
for i in range(5):
	model.add(Dense(4, activation='elu'))
model.add(Dense(1))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.fit(x, y, epochs=200, batch_size=150, verbose=1)
print()

for i in [0, 10, 20, 30, 50, 2025, 2750, 7507]:
	print(x[i], model.predict(np.array([x[i]])), y[i])

print(model.get_weights())


# optimizers:
	# ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
	# sgd: 0.94
	# rmsprop: 0.92
	# adam: 0.86
	# adadelta: 2.59
	# adagrad: 1.41
	# adamax: 1.00
	# nadam: 1.74
	# ftrl: 2.61

# loss:
	# ['mean_absolute_error', 'logcosh', 'mean_squared_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error']
	# mean_absolute_error: 0.86
	# logcosh: 0.87
	# mean_squared_error: 1.66
	# mean_absolute_percentage_error: 3.13
	# mean_squared_logarithmic_error: 1.11

# activations:
	# ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
	# relu 0.87
	# sigmoid: 0.98
	# softmax: 1.66
	# softplus: 0.86
	# softsign: 0.79
	# tanh: 0.84
	# selu: 0.86
	# elu: 0.71
	# exponential: nan

# layers
	# 0: 1.08
	# 1: 0.89
	# 2: 0.86
	# 3: 0.69, best: 0.59
	# 4: 0.77
	# 5: 0.65
	# 6: 0.62
	# 7: 0.72
	# 8: 0.60
	# 9: 0.63
	# 10: 0.52
	# 11: 0.62
	# 12: 0.79
	# 13: 0.61
	# 14: 0.50
	# 15: 0.67
	# 16: 0.61
	# 17: 0.48
	# 18: 0.66
	# 19: 0.63

# count of neurons:
	# 64, optial layers >= 4