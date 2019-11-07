import joblib
from os import path,listdir
import os, fnmatch
from keras.models import model_from_json
import json


Model_PATH = path.join('Model')
Model_Accuracy_jsonfile = "modelAcc.json"    

def getModelList():
    global Model_PATH
    ModelFilesList = []
    Allfiles = listdir(Model_PATH)
    for afile in Allfiles:
        if ".joblib" in afile or ".h5" in afile:
            ModelFilesList.append(afile)
    return ModelFilesList

def SaveModel(name):
    pass

def getModel(filename):
    global Model_PATH
    filename = path.join(Model_PATH,filename)
    print ("Loading Model : ",filename,"\n...")
    if ".joblib" in filename:
        return joblib.load(filename)
    elif ".h5" in filename:
        return loadKerasModel(filename)
        
def TrainModel():
    pass

def loadKerasModel(filename):
    jsonName = filename[:-3]+'.json'
    # load json and create model
    with open(jsonName, 'r') as json_file: 
        model_json = json_file.read()
        myModel = model_from_json(model_json)
    # load weights into new model
    myModel.load_weights(filename)
    return myModel

def loadModelAccfromFile(modelName):
    global Model_Accuracy_jsonfile
    global Model_PATH
    filepath = path.join(Model_PATH,Model_Accuracy_jsonfile)
    with open(filepath, 'r') as accfile:
        Acc_dict = json.load(accfile)
        # print(Acc_dict)
        # print(type(Acc_dict))
    if modelName in Acc_dict:
        return Acc_dict[modelName]
    else:
        print("Model Acc records not found. Makeing a new entry")
        acc = {"BY_MERGE":"NA","BY_CLASS":"NA","BALANCED":"NA","DIGITS":"NA","LETTERS":"NA","MNIST":"NA"}
        SaveModelAccTofile(modelName,acc)
        return acc



def SaveModelAccTofile(modelname,Mode_acc_dict):
    global Model_Accuracy_jsonfile
    global Model_PATH
    filepath = path.join(Model_PATH,Model_Accuracy_jsonfile)
    with open(filepath, 'r') as accfile:
        Acc_dict = json.load(accfile)
    Acc_dict[modelname] = Mode_acc_dict
    with open(filepath, 'w') as accfile:
        json.dump(Acc_dict,accfile)

