from Data.Data import Data, DatSet
from Model.modelUtils import getModelList, getModel
import Model.modelUtils as M_Util
from Model.Model import Model,KS_Model
import numpy as np
from random import sample
from Image.imageutils import getImageList, process_Image
from math import sqrt,ceil
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

_myData = None 
_myModel = None
def main():
	global _myData
	global _myModel
	while True:
		currentDataset = _myData.dataSetName if hasattr(_myData,"dataSetName") else "-"
		currentModel = _myModel.modelName if hasattr(_myModel,"modelName") else "-"
		currentAcc = _myModel.Acc if hasattr(_myModel,"Acc") else 0.0
		msg = "\nSelect Function \n 1. Load Data ({0}) \n 2. Load Model ({1}) \n 3. Predict\n 4. Evaluate (Accuracy:{2:5.2f}%)\n 5. Train Model\n (Type 'exit' to quit) \n>".format(currentDataset,currentModel,currentAcc)
		fun = input(msg)
		if fun == "exit":
			break
		elif fun == "Load Data" or fun == "1":
			_myData = LoadDataSet()
			continue
		elif fun == "Load Model" or fun == "2":
			_myModel = LoadModel()
			continue
		elif fun == "Predict" or fun == "3":
			if (_myData != None and _myModel != None):
				Predict()
			else:
				print("please Load DataSet and Model first")
				input("Press Enter to continue...")
			continue
		elif fun == "Evaluate" or fun == "4":
			if (_myData != None and _myModel != None):
				Evaluate()
			else:
				print("please Load DataSet and Model first")
				input("Press Enter to continue...")
			continue
		elif fun == "Train Model" or fun == "5":
			print("Not Implemented Yet")
			continue
		elif fun == "Test" or fun == "test":
			Test()
			continue
			

def LoadDataSet():
	global _myModel
	global _myData
	message = "\nChose a Dataset \n 1. BY_MERGE \n 2. BY_CLASS \n 3. BALANCED \n 4. DIGITS \n 5. LETTERS \n 6. MNIST \n>"
	userInput = input(message)
	if hasattr(_myModel,"Acc"):
		_myModel.resetCurrentAcc()
	if userInput == "1" or userInput == "BY_MERGE":
		return Data(DatSet.BY_MERGE)
	elif userInput == "2" or userInput == "BY_CLASS":
		return Data(DatSet.BY_CLASS)
	elif userInput == "3" or userInput == "BALANCED":
		return Data(DatSet.BALANCED)
	elif userInput == "4" or userInput == "DIGITS":
		return Data(DatSet.DIGITS)
	elif userInput == "5" or userInput == "LETTERS":
		return Data(DatSet.LETTERS)
	elif userInput == "6" or userInput == "MNIST":
		return Data(DatSet.MNIST)

def LoadModel():
	global _myData
	global _myModel
	modelList = getModelList()
	message = "Chose a Model \n"
	i = 1
	for model in modelList:
		message = message + str(i) +". "+ model +"\n"
		i +=1
	userInput = int(input(message+">"))
	if userInput <= len(modelList):
		name = modelList[userInput-1]
		model = getModel(name)
		if "keras.engine.sequential" in str(type(model)):
			return KS_Model(model,name=name,mapping="emnist-bymerge-mapping.txt")
		else :
			return Model(model,name=name)
	else:
		print("Invalid input select between 1 -",len(modelList))
		input("Press Enter to continue...")
		LoadModel()

def Predict():
	global _myData
	global _myModel
	message = "Predict from \n 1. Test Set \n 2. Image file \n"
	userInput = int(input(message+">"))
	if userInput == 1:
		Predict_from_testSet()
	elif userInput == 2:
		Predict_from_imageFile()

def Predict_from_imageFile():
	imageList = getImageList()
	message = "Chose a Image \n"
	i = 1
	for image in imageList:
		message = message + str(i) +". "+ image +"\n"
		i += 1
	userInput = int(input(message+">"))
	if userInput <= len(imageList):
		image = imageList[userInput-1]
		charList = process_Image(image)
		reslist = []
		for char in charList:
			res = _myModel._Predict(char)
			reslist.append(res)
		drawImages(charList,reslist)
		print ("Predicted output is ",reslist)
	else :
		print(image," : Image not found")
	

def Predict_from_testSet():
	global _myData
	global _myModel
	message = "Choose a Data Point Between 0-"+str(_myData.test_x.shape[0])+" from the Test set \n"
	userInput = int(input(message+">"))
	if userInput < _myData.test_x.shape[0]:
		res = _myModel._Predict(np.copy(_myData.test_x[userInput]))
		lbl = _myData.test_y[userInput]
		lbl = _myData.mapping[lbl]
		_myData.drawImage(userInput,res=res,lbl=lbl)
		print ("Predicted output is ",res)
		print ("lbl output is ",lbl)
		return res
	else :
		Predict()

def Evaluate():
	global _myData
	global _myModel
	message = "Get model Accuracy from \n 1. Evaluate New \n 2. Load from File \n"
	userInput = int(input(message+">"))
		
	if userInput == 1 :
		out = _myData.getTestOutputList() 
		if "IS_FIVE" in _myModel.modelName:
			out = [str(i == "5") for i in out]
		acc = _myModel.evaluateAccuracy(_myData.test_x,out)
		_myModel.saveAccuracy(_myData.dataSetName,acc)
	elif userInput == 2:
		acc = _myModel.getAccuracy(_myData.dataSetName)
		if acc == -1:
			print("Model Accuracy not Found. Evaluate the Model again to Find Accuracy..")
			input("Press Enter to continue...")
			return None
	print ("Model : ",_myModel.modelName," is ",acc," percent accurate on ",_myData.dataSetName," test set.\n")
	showSampleImages()
	return acc

def showSampleImages():
	global _myData
	global _myModel
	if len(_myModel.correctIndex) >= 8 and len(_myModel.wrongIndex)>=8:
		indexList = sample(_myModel.correctIndex,8)+sample(_myModel.wrongIndex,8)
	else :
		indexList = sample(range(0,_myData.test_x.shape[0]),16)
	
	predList = [_myModel._Predict(np.copy(_myData.test_x[index])) for index in indexList]
	
	outList = _myData.getTestOutputList()
	if "IS_FIVE" in _myModel.modelName:
		print("Binary Classifier model")
		expectOutList = [str(outList[int(index)] == "5") for index in indexList]
	else:
		expectOutList = [outList[int(index)] for index in indexList]
	
	_myData.drawImages(indexList,predOut=predList,expectedOut=expectOutList)
	

def Status():
	global _myData
	global _myModel
	if hasattr(_myData,"dataSetName"):
		print("Current DataSet : ",_myData.dataSetName)
	else :
		print("Current DataSet : None/Unknown")
	if hasattr(_myModel,"modelName"):	
		print("Current Model : ",_myModel.modelName)
	else:
		print("Current Model : None/Unknown")
	if hasattr(_myModel,"Acc"):	
		print("Accuaracy : ",_myModel.Acc)
	else :
		print("Accuaracy : None/Unknown")

	
def Test():
	global _myData
	global _myModel
	message = "Choose a output"
	userInput = input(message+">")
	res = _myModel.find_Indexes_Matching_Output(_myData.test_x,userInput)
	print(res)
    
def drawImages(charList,predOut):
	for i in range(len(charList)):
		n = ceil(sqrt(len(charList)))
		plt.subplot(n,n,i+1)
		img = charList[i]
		if i < len(predOut): 
			title_obj  = plt.title("Out:"+str(predOut[i]))
		plt.imshow(img.reshape([28, 28]), cmap = mpl.cm.binary)
		plt.axis('off')
	plt.subplots_adjust(hspace=0.6)
	plt.savefig('Data/img.png')
	image = Image.open('Data/img.png')
	image.show()

if __name__ == '__main__':
	main()
	