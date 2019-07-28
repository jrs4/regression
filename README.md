# Objective
Develop an artifical neural network in Tensorflow / Keras that can predict values for protein and moisture in wheat samples with a a maximum permissible error of 0.4%. The spectra files contains the absorbance data of wheat samples ranging from 720 to 1100nm in 10nm intervals (38 pixels) along with the laboratory values for both protein and moisture 

# Regression Model.ipynb - Wheat Calibration

There are 3 sets of folders:
* (1) Training: contains the training set, data from the master instrument 
* (2) Validation: contains the validation set, data from the master instrument
* (3) Testing: contains the testing set from different instruments. Because each system has its own set of slope and biases, there can only be one test set at any time. 

## Training Data pre-treatment:
* Load calibration data for training
* Only use pixels from 9 to 33 (25 pixels) for training
* Drop any rows with y values outside limits (e.g. 5-20% for protein in wheat)
* Fit the training data into Principal Component Analysis (PCA) from scikit-learn
* Transform the training data into principal components
* Use MinMaxScaler() to fit the data between (-1, 1)

## Validation Data pre-treatment:
* Load validation data
* Only use pixels from 9 to 33 (25 pixels) for training
* Drop any rows with y values outside limits (e.g. 5-20% for protein in wheat)
* Transform the validation data into principal components using the PCA model from the training set
* Use MinMaxScaler() to fit the data between (-1, 1)
* Train Model in TensorFlow / Keras:

After the data is ready for the neural network, create the model in Keras, compile it and fit it.

## Testing Data pre-treatment:
* Same as validation data pre-treatment
* Check predictions and calculates slopes and biases

## Saving the models:
* Pickle PCA algorithm from the training set
* Pickle MinMaxScaler algorithm from the training set
* Save models for protein and moisture in HDF5 format (.h5)

# Prediction.ipynb
* Reads the pickled algorithms and the keras models
* Applies the same pre-treatment in the data using the unpickled models
* Makes inferences and calculates slopes and biases