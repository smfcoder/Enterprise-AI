from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import numpy

dataset = numpy.loadtxt("featureVectors.csv", delimiter=",")
# split into input (X) and output (Y) variables
# Train Set
X = dataset[:, 0:10]
y_int = dataset[:, 10:]
Y = to_categorical(y_int)

# Test Set
XTest = dataset[0:200, 0:10]
ytest_int = dataset[0:200, 10:]
yTest = to_categorical(ytest_int)
# create model
model = Sequential()
model.add(Dense(output_dim=100, init='uniform', activation='relu'   , input_dim=10))
model.add(Dense(output_dim=100 , init='uniform', activation='relu'   , input_dim=100))
model.add(Dense(output_dim=3   , init='uniform', activation='softmax', input_dim=100))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X, Y, nb_epoch=25, batch_size=10)
# evaluate the model
scores = model.evaluate(XTest, yTest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Train Set
X = dataset[:2900, 0:10]
y_int = dataset[:2900, 10:]
Y = to_categorical(y_int)
# Test Set
XTest = dataset[2900:, 0:10]
ytest_int = dataset[2900:, 10:]
yTest = to_categorical(ytest_int)
# Fit the model
model.fit(X, Y, nb_epoch=25, batch_size=10)
# evaluate the model
scores = model.evaluate(XTest, yTest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


# #Only code needed to save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
##################################