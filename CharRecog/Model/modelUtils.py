import joblib
from os import path,listdir
import os, fnmatch
import json
import time
import matplotlib.pyplot as plt 

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau


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

def SaveKarasModel(model,name):
    global Model_PATH
    fileName =  path.join(Model_PATH,name)
    json = model.to_json()
    with open(fileName+".json", "w") as json_file:
        json_file.write(json)
    model.save_weights(fileName+".h5")

def getModel(filename):
    global Model_PATH
    filename = path.join(Model_PATH,filename)
    print ("Loading Model : ",filename,"\n...")
    if ".joblib" in filename:
        return joblib.load(filename)
    elif ".h5" in filename:
        return loadKerasModel(filename)
        

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


def TrainModel(myData,numClass,batch_size=86,epochs=30,learningRate=0.001):
    if (myData.dataSetName == "LETTERS"): # Letters dataset starts from 1 instead of zero
        numClass += 1
    model = GetCNNmodel(learningRate,numClass)

    myData.train_x = myData.train_x / 255.0
    myData.train_x = myData.train_x.reshape(-1,28,28,1)
    
    myData.train_y = to_categorical(myData.train_y, num_classes = numClass)

    myData.test_x = myData.test_x / 255.0
    myData.test_x = myData.test_x.reshape(-1,28,28,1)
    myData.test_y = to_categorical(myData.test_y, num_classes = numClass)

    # With data augmentation to prevent overfitting 

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


    datagen.fit(myData.train_x)
    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
    # Fit the model
    history = model.fit_generator(datagen.flow(myData.train_x,myData.train_y, batch_size=batch_size),
                              epochs = epochs, validation_data = (myData.test_x,myData.test_y),
                              verbose = 2, steps_per_epoch=myData.train_x.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
    print("Done .........!")
    name = "model_CNN_"+str(myData.dataSetName)+str(time.time())
    SaveKarasModel(model,name)
    plot_history(history)

    return name,model

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [Thousand Dollars$^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
    plt.legend()
    plt.ylim([0,50])
    plt.show()   

def GetCNNmodel(learningRate,numClass):
    # Set the CNN model 
    #CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                    activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(numClass, activation = "softmax"))

    # Define the optimizer
    optimizer = RMSprop(lr=learningRate, rho=0.9, epsilon=1e-08, decay=0.0)
    # Compile the model
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    return model

