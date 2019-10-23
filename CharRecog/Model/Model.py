from Data.dataUtil import LoadMapping
from Model.modelUtils import SaveModelAccTofile, loadModelAccfromFile
from Data.Common import SIMILAR_CHARS,SIMILAR_CHAR_GRP
import numpy as np


class Model:
    def __init__(self,model,name):
        self.correctIndex = []
        self.wrongIndex = []
        self.Acc = 0
        self.model = model
        if name != None:
            self.modelName = name

    def _Predict(self,x):
        y = self.model.predict([x])
        return str(y[0])

    def find_Indexes_Matching_Output(self,test_set,output):
        res = []
        for i in range(test_set.shape[0]):
            x = np.copy(test_set[i])
            if self._Predict(x) == output:
                res.append(i+1)
        return res
    
    def evaluateAccuracy(self,test_set,expectedOutput):
        print("Wait while we evalute Model ",self.modelName,"...")
        self.resetCurrentAcc()
        similerCount = 0
        for i in range(test_set.shape[0]):
            x = np.copy(test_set[i])
            pred = self._Predict(x)
            expect = expectedOutput[i]
            if pred == expect or pred.lower() == expect.lower():
                self.correctIndex.append(i)
            elif pred in SIMILAR_CHARS and expect in SIMILAR_CHAR_GRP[pred]:
                similerCount += 1
                self.wrongIndex.append(i)
            else :
                self.wrongIndex.append(i)

        self.Acc = (len(self.correctIndex)*100 + similerCount/2)/test_set.shape[0]
        return self.Acc


    def resetCurrentAcc(self):
        self.correctIndex = []
        self.wrongIndex = []
        self.Acc = 0.0

    def saveAccuracy(self,dataSet,val):
        acc_dict = loadModelAccfromFile(self.modelName)
        if dataSet in acc_dict:
            acc_dict[dataSet] = self.Acc
        SaveModelAccTofile(self.modelName,acc_dict)   
        
    def getAccuracy(self,dataSet):
        acc_dict = loadModelAccfromFile(self.modelName)
        acc = acc_dict[dataSet]
        if acc != "NA":
            self.Acc = float(acc)
        else:
            return -1
        return self.Acc


class KS_Model(Model):
    def __init__(self,model,name=None,mapping = None):
        super().__init__(model,name)
        if mapping == None:
            self.mapping = "emnist-bymerge-mapping.txt"
        else :
            self.mapping = mapping 

    def _Predict(self,x):
        x = x/255.0
        x = x.reshape(-1,28,28,1)
        y = self.model.predict(x)
        mymap = LoadMapping(self.mapping)
        y = mymap[int(np.argmax(y,axis = 1))]
        return y

