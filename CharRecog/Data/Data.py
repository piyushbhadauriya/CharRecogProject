from enum import IntEnum
import numpy as np
from Data.dataUtil import LoadData,LoadMapping
from Data.Common import SIMILAR_CHARS,SIMILAR_CHAR_GRP
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

class DatSet(IntEnum):
    BY_MERGE = 0
    BY_CLASS = 1
    BALANCED = 2
    DIGITS = 3
    LETTERS = 4
    MNIST = 5

## OUTER : DataSets
## inner : [train,test,mapping]
_DataSetList = [["emnist-bymerge-train.csv","emnist-bymerge-test.csv","emnist-bymerge-mapping.txt"],
            ["emnist-byclass-train.csv","emnist-byclass-test.csv","emnist-byclass-mapping.txt"],
            ["emnist-balanced-train.csv","emnist-balanced-test.csv","emnist-balanced-mapping.txt"],
            ["emnist-digits-train.csv","emnist-digits-test.csv","emnist-digits-mapping.txt"],
            ["emnist-letters-train.csv","emnist-letters-test.csv","emnist-letters-mapping.txt"],
            ["emnist-mnist-train.csv","emnist-mnist-test.csv","emnist-mnist-mapping.txt"]]

class Data:
    def __init__(self,dataSet):
        self.dataSetName = dataSet.name
        global _DataSetList
        print("Loading Training Set ...")
        self.train_y,self.train_x = LoadData(_DataSetList[dataSet][0])
        print("Loading Test Set ...")
        self.test_y,self.test_x  = LoadData(_DataSetList[dataSet][1])
        print("Load Mapping ...")
        self.mapping = LoadMapping(_DataSetList[dataSet][2])
        print("Done!")
        self.displaySummery()
    
    def displaySummery(self):
        unique, counts = np.unique(self.train_y, return_counts=True)
        mydict = dict(zip(unique, counts))
        total = 0
        print("Dataset-",self.dataSetName)
        print("Char : Count")
        for key,value in mydict.items():
            print(self.mapping[int(key)]," : ",value)
            total += value
        print("----------------------")
        print("Total : ",total)
    
    def getTestOutputList(self):
        if not hasattr(self,"mapped_test_y"):
            f = lambda x: self.mapping[x]
            expectedOut = []
            for i in range(self.test_y.shape[0]):
                expectedOut.append(f(self.test_y[i]))
            self.mapped_test_y = expectedOut
        return self.mapped_test_y

    def getTrainOutputList(self):
        if not hasattr(self,"mapped_train_y"):
            f = lambda x: self.mapping[x]
            expectedOut = []
            for i in range(self.train_y.shape[0]):
                expectedOut.append(f(self.train_y[i]))
            self.mapped_train_y = expectedOut
        return self.mapped_train_y

    def drawImage(self,index,res=None,lbl=None):
        plt.figure()
        if type(index) == int:
            img = self.test_x[index]
        else:
            img = index
        if res != None: 
            plt.title("Predicted Out : "+str(res)+"\nLabel : "+str(lbl))
        plt.imshow(img.reshape([28, 28]), cmap = mpl.cm.binary)
        plt.savefig('Data/img.png')
        image = Image.open('Data/img.png')
        image.show()
        
    
    def drawImages(self,indexList,predOut,expectedOut=[]):
        for i in range(len(indexList)):
            plt.subplot(4,4,i+1)
            index = indexList[i]
            img = self.test_x[index]
            lbl = self.test_y[index]
            lbl = self.mapping[lbl]
            expect = str(expectedOut[i]) if i < len(expectedOut) else lbl
            if i < len(predOut): 
                title_obj  = plt.title("Out : "+str(predOut[i])+"\nLabel : "+str(lbl))
            plt.imshow(img.reshape([28, 28]), cmap = mpl.cm.binary)
            plt.axis('off')
            print("index:",index," Pred:",predOut[i]," Expect:",expect)
            if str(predOut[i]) == expect:
                plt.setp(title_obj, color='g')
            elif predOut[i].lower() == expect.lower():
                plt.setp(title_obj, color='lime')
            elif predOut[i] in SIMILAR_CHARS and expect in SIMILAR_CHAR_GRP[predOut[i]]:
                plt.setp(title_obj, color='y')
            else:
                plt.setp(title_obj, color='r')
        plt.subplots_adjust(hspace=0.6)
        plt.savefig('Data/img.png')
        image = Image.open('Data/img.png')
        image.show()


    