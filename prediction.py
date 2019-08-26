import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import time
import sys

pkl_filename = "./Model/pickle_pca_model.pkl"
with open(pkl_filename, 'rb') as file:
    pca = pickle.load(file)
    
pkl_filename = "./Model/pickle_scaler_model.pkl"
with open(pkl_filename, 'rb') as file:
    scaler = pickle.load(file)
    
model1 = load_model('./Model/Protein.h5')
model2 = load_model('./Model/Moisture.h5')
Cal1Name = "Protein"
Cal2Name = "Moisture"
Cal3Name = "Oil"

#read adjustments (slopes and biases for each constituent)
adjustments = [1,0,1,0,1,0]
try:
    adjustments = np.loadtxt("./Model/adjustments.csv", delimiter=",")
except:
    pass

#a way to terminate the process from the analysis software
def StopProcess():
    if os.path.isfile("./stop.txt"):
        os.remove("./stop.txt")
        return True
    return False

#keep in an infinite loop until the analysis software stops 
#so we don't need to load tf and the models all the time
while not StopProcess():
    #read file containing spectra for inference
    while not os.path.isfile("./predict.csv"):
        time.sleep(0.01)
        if StopProcess():
            sys.exit()

    df_test = pd.read_csv("./predict.csv", index_col=None, header=0)

    #rename "cal1" "cal2" "cal3"
    df_test = df_test.rename(index=str, columns={"Cal1": Cal1Name, "Cal2": Cal2Name, "Cal3": Cal3Name})
    df_test = df_test.rename(index=str, columns={"cal1": Cal1Name, "cal2": Cal2Name, "cal3": Cal3Name})

    #Drop values of spectra if NaN
    for i in range(9,33):
        df_test = df_test.dropna(subset=["Sp{}".format(i)])
        
    #Drop values of protein if NaN
    df_test = df_test.dropna(subset=[Cal1Name])

    test_spectra = df_test.loc[:,"Sp9":"Sp33"].values
                            
    test_values = [1,2,3]
    if Cal1Name in df_test:
        test_values[0] = df_test[Cal1Name].values
    if Cal2Name in df_test:
        test_values[1] = df_test[Cal2Name].values
    if Cal3Name in df_test:
        test_values[2] = df_test[Cal3Name].values

    test_spectra_pca = pca.transform(test_spectra)

    test_spectra_pca = scaler.transform(test_spectra_pca)

    prediction = np.array([1.0,2.0])
    prediction[0] = model1.predict(test_spectra_pca)*adjustments[0]+adjustments[1]
    prediction[1] = model2.predict(test_spectra_pca)*adjustments[2]+adjustments[3]

    print("Lab Values {} = {}".format(Cal1Name, test_values[0]))
    print("Prediction {} = {}\n".format(Cal1Name, np.around(prediction[0].reshape(test_values[0].shape[0]),decimals=1)))
    print("Lab Values {} = {}".format(Cal2Name, test_values[1]))
    print("Prediction {} = {}".format(Cal2Name, np.around(prediction[1].reshape(test_values[1].shape[0]),decimals=1)))
    
    np.savetxt("./results.csv", prediction, delimiter=",")
    os.remove("./predict.csv")