# Load liabraries
import math 
import csv
import random
import os
import shutil
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras import utils
from keras import models
from keras import layers
from keras import metrics
from keras import optimizers
from Data_Manager import DataSets_Maker
from Data_Manager import Noise_Management
import matplotlib.pyplot as plt
#######################################################################################################
class Function_Classifier():
    DropOut=False
    DropOut_Rate=0
    RootPath=os.getcwd()
    NFunctions=11###############################
    ActiveProject=False
#######################################################################################################
#######################################################################################################
############################## NPLC = Non Pre Learning Classifier #####################################
#######################################################################################################
#######################################################################################################
    def _MaxOfRow(arr):#find maximum value in a one row array and its Index
        n=len(arr)
        Max=0
        Index=0
        for i in range(n):
            if arr[i]>Max : 
                Max=arr[i]
                Index=i
        return(Max, Index)
#######################################################################################################        
    def NonPrelearning_Classifier(path):#making new model for classification without prelearning functions
        os.chdir(path)
        modelname = input('Enter New NonPreLearning Model Name(0 for Back) : ')
        if modelname=='0' : return(0)
        os.mkdir(modelname)
        os.chdir(modelname)
        modelpath=os.getcwd()
        NFiles = int(input('Enter Number of Files(Data) Should be Created :'))
        NDatainFiles = int(input('Enter Number of Data Sould be Created in Every File :'))
        Iteration = int(input('Enter Number of Iteration You Want Use For Training : '))
        NoiseName, NoisePerc = Function_Classifier.Data_Maker_4_NPLC(NFiles, NDatainFiles)
        x_train, y_train, x_test, y_test = Function_Classifier.LoadNetData_4_NPLC(modelpath, NFiles, NDatainFiles)
        Model = Function_Classifier.RunNNForNPLC(x_train, y_train, x_test, y_test, NFiles, NDatainFiles, Iteration)
        os.chdir(path)
        os.chdir(modelname)
        NPLCModelINformation=list()###################Saving Model
        NPLCModelINformation.append(NFiles)#### Number of Files
        NPLCModelINformation.append(NDatainFiles)#### Number of Data point in every file
        NPLCModelINformation.append(Iteration)#### Number of Iteration
        NPLCModelINformation.append(Function_Classifier.DropOut)#### True or False
        NPLCModelINformation.append(Function_Classifier.DropOut_Rate)#### Rate of Drop out
        NPLCModelINformation.append(NoiseName)# Noise Add to Train Dataset
        NPLCModelINformation.append(NoisePerc)# Percent of Nois Data in Train Dataset
        NPLCInfoFileName = modelname + 'Info.csv'
        ModelFileName = NPLCInfoFileName.replace('Info.csv','.h5')
        FH = open(NPLCInfoFileName,'w',newline='')
        Writer=csv.writer(FH)
        Writer.writerow(NPLCModelINformation)
        FH.close()
        Model.save(ModelFileName)
#######################################################################################################
    def Data_Maker_4_NPLC(NFiles, NDatainFiles):#making data for classification without preleaning 6
        root = os.getcwd()
        for i in range(1,12):
            os.mkdir(Function_Classifier.Functions_Name(i))
            os.chdir(Function_Classifier.Functions_Name(i))
            for j in range(NFiles):
                if j<10  : P = '-00' 
                else : P = '-0'
                FileName = Function_Classifier.Functions_Name(i) + P + str(j) + '.csv'
                print('Creating Files of ',Function_Classifier.Functions_Name(i),': ',FileName,'           ',end='\r')
                FunctionList = Function_Classifier.DataList_Maker(i, NDatainFiles)
                FH = open(FileName, 'w', newline='')
                writer = csv.writer(FH)
                writer.writerows(FunctionList)
                FH.close()
            os.chdir(root)
        NoiseName, NoisePerc = Function_Classifier.AddNoise_NPLC(root)
        return(NoiseName, NoisePerc)
#######################################################################################################
    def LoadNetData_4_NPLC(modelpath, NFiles, NDatainFiles):#load data for classification without prelearning to use in neural net
        os.chdir(modelpath)
        x_train = list()
        y_train = list()
        x_test = list()
        y_test = list()
        for i in range(1,12):#Counting Folders
            os.chdir(Function_Classifier.Functions_Name(i))
            Fdata = list()
            for j in range(NFiles):#Counting Files in every Folder
                filedata = list()
                if j<10 : P ='-00'
                else : P = '-0'
                FileName = Function_Classifier.Functions_Name(i) + P + str(j) + '.csv'
                print('Loading Files of ',Function_Classifier.Functions_Name(i),': ',FileName,'            ',end='\r')
                FH = open(FileName)
                Reader = csv.reader(FH)
                filedata = list(Reader)
                for k in range(NDatainFiles):#Counting record in every File
                    filedata[k][0]=float(filedata[k][0])
                    filedata[k][1]=float(filedata[k][1])
                Fdata.append(filedata)
                FH.close()
            for k in range(NFiles):
                x_train.append(Fdata[k])
                y_train.append(i-1)
            NTest=Function_Classifier.Round_Percentage(NFiles,25)
            os.chdir(modelpath)
        NTest=Function_Classifier.Round_Percentage(NFiles,25)
        x_test, y_test , NN, NP = Function_Classifier.Making_TestDataSet_NPLC(NTest, NDatainFiles, False)
        xtr=np.array(x_train)
        ytr=np.array(y_train)
        xte=np.array(x_test)
        yte=np.array(y_test)
        return(xtr, ytr, xte, yte)
#######################################################################################################
    def RunNNForNPLC(x_train, y_train, x_test, y_test, NFile, NDatainFiles, Iteration): 
        #start neural network
        print("\nStarting Neural Network...\n")
        network = models.Sequential() 
        ############################################################################################################### DropOut ##################################################################################################################
        if Function_Classifier.DropOut :
            network.add(layers.Dropout(Function_Classifier.DropOut_Rate, input_shape=(NDatainFiles,2)))
        network.add(layers.Dense(units=64, activation="relu", input_shape=(NDatainFiles,2)))###Layer1
        network.add(layers.Dense(units=32, activation="relu"))
        network.add(layers.Dense(units=32, activation="relu"))
        network.add(layers.Dense(units=16, activation="relu"))
        network.add(layers.Flatten())
        network.add(layers.Dense(units=11, activation="sigmoid"))###Layer 5
        #compile neural network
        network.compile(loss="sparse_categorical_crossentropy", #Mean squared error
                        optimizer="adam", #optimization algorithm
                        metrics=["accuracy"])
        #Train neural network
        max=0
        min=100
        for i in range(Iteration):
            print("epochs #: ",i,'/',Iteration,end='')
            history = network.fit(x_train, #features
                                  y_train, #Target vector
                                  epochs=1, #number of echos
                                  verbose=0,# No output
                                  batch_size=1, #number of observation per batch
                                )#Test data
            prediction = network.predict([x_train],batch_size=1)
            scores = network.evaluate(x_train, y_train, verbose=0)
            print('\tScore is : ',scores)
            if scores[0]>max : max=scores[0]
            if scores[0]<min : min=scores[0]
        testprediction=network.predict([x_test],batch_size=1)
        Success=0
        for i in range(len(y_test)):
            Max , Index = Function_Classifier._MaxOfRow(testprediction[i])
            print('Model Prediction is :',Function_Classifier.Functions_Name(Index+1),'with',round(Max*100,2),'%','and the true answer is :',Function_Classifier.Functions_Name(y_test[i]+1)) 
            if (Index==y_test[i]):
                Success+=1
        print('\n\nFrom ',len(y_test),' Test Data, Our Model''s Correct Prediction was',Success,
               'and Success Percentage is :',round((Success/len(y_test))*100,2),'%')
        input('\n\nPress Enter To Continue...')
        return(network)
#######################################################################################################        
    def AddNoise_NPLC(Path):# Adding Noise To a Dataset For NPLC Projects
        Noise_Name=0
        Perc=0
        ans = input('Do You Want to Add Noise To Test Data(y/n)? ')
        if ans!='y' and ans!='Y' : return(Noise_Name, Perc)
        print('\nHere are Noises You Can Choose For Adding to Data Sets:\n\n1- Gaussian Noise\n2- Rayleigh Noise'
              '\n3- Gamma Noise\n4- Exponential Noise\n5- Uniform Noise\n\nPlease Enter The Number of The Noise You Want '
              'Add To Data Sets: ',end='') 
        Noise_Name=int(input())
        if 0<Noise_Name<6:
            print('Please Enter The Percentage of Noise You Want to Add(1-100) : ',end='')
            Perc=int(input())
        else: return(Noise_Name, Perc)
        if not(0<Perc<100) : return(Noise_Name, Perc)    
        for i in range(Function_Classifier.NFunctions):
            os.chdir(Path)
            os.chdir(Function_Classifier.Functions_Name(i+1))
            Nfiles = len(os.listdir())
            for j in range(Nfiles):
                if j<10  : P = '-00' 
                else : P = '-0'
                FileName = Function_Classifier.Functions_Name(i+1) + P + str(j) + '.csv'
                FH = open(FileName)#######>>> Reading File Content
                Reader = csv.reader(FH)
                filedata = list(Reader)
                FH.close()
                for k in range(len(filedata)):
                    filedata[k][0]=float(filedata[k][0])
                    filedata[k][1]=float(filedata[k][1])
                filedata = Noise_Management.Add_Noise(filedata, Noise_Name, Perc)#####>>> Adding noise tO Data
                FH = open(FileName,'w', newline='')
                Writer = csv.writer(FH)
                Writer.writerows(filedata)#####>>>> Rewrite The File With Noisy Data
                FH.close()
        return(Noise_Name, Perc)
#######################################################################################################
    def Use_NPLC_Model(Model, ModelName) : #Use A Saved Model Of NPLC and Use new Test Dataset For Evalute The Model
        os.chdir(ModelName)
        FH = open(ModelName+'Info.csv')
        Reader = csv.reader(FH)
        Modelinformation = list(Reader)
        ModelInfo = list()###>>change model info to usable data(such as int)
        for i in range(3):
            ModelInfo.append(int(Modelinformation[0][i]))
        if (Modelinformation[0][3]=='True') : ModelInfo.append(True)#Drop Out
        else : ModelInfo.append(False)
        ModelInfo.append(float(Modelinformation[0][4]))# Drop Out Rate
        Model.summary()#####################################################
        print('The Model was Trained by',ModelInfo[0],'Data Files for every Function',
               'wich every File Contains',ModelInfo[1],'Data.',
               '\nThe Model Trained in',ModelInfo[2],'Iteration.',end="")
        if (ModelInfo[3]) : print('The Model has Drop Out Layer with the Rate of ',ModelInfo[4])
        else : print('The Model Has no Drop Out Layer.')
        NumData = int(input('\nPlease Enter The Number of Test Data for Evalution of Model : '))
        x_test , y_test, NoiseName, NoisePerc = Function_Classifier.Making_TestDataSet_NPLC(NumData,ModelInfo[1], True)
        testprediction = Model.predict([x_test],batch_size=1)
        Scores = Model.evaluate(x_test, y_test, verbose=0)
        Success=0
        ch = input('Do You Want to see Every Test Result(y/n)? ')
        if ch=='y' or ch=='Y' : show=True
        else : show=False
        for i in range(len(y_test)):
            Max , Index = Function_Classifier._MaxOfRow(testprediction[i])
            if (show) : print('Model Prediction is :',Function_Classifier.Functions_Name(Index+1),'with',round(Max*100,2),'%','and the true answer is :',Function_Classifier.Functions_Name(y_test[i]+1)) 
            if (Index==y_test[i]):
                Success+=1
        print('\n\nFrom ',len(y_test),' Test Data, Our Model\'s Correct Prediction was',Success,
               '\nand Success Percentage is :',round((Success/len(y_test))*100,2),'%')
        if NoiseName==0 : print('\nThese Test Dataset Have no Noise.')
        else : print('These Test Dataset Have',NoisePerc,'% Noise of ',
        Noise_Management.Noise_Name(NoiseName),'Noise.')
        print('\nThe Score of Model is : ',Scores,'\n\nPress Enter To Continue...',end='')
        input()
#######################################################################################################
    def Making_TestDataSet_NPLC(NFiles, NDatainFiles, Noise):
        path = os.getcwd()
        os.mkdir('TestDataSet')
        os.chdir('TestDataSet')
        root = os.getcwd()
        for i in range(1,12):####Creating...
            os.mkdir(Function_Classifier.Functions_Name(i))
            os.chdir(Function_Classifier.Functions_Name(i))
            for j in range(NFiles):
                if j<10  : P = '-00' 
                else : P = '-0'
                FileName = Function_Classifier.Functions_Name(i) + P + str(j) + '.csv'
                print('Creating Test Data...',round((i/22)*100, 2),'%              ',end='\r')
                FunctionList = Function_Classifier.DataList_Maker(i, NDatainFiles)
                FH = open(FileName, 'w', newline='')
                writer = csv.writer(FH)
                writer.writerows(FunctionList)
                FH.close()
            os.chdir(root)
        if Noise : NoiseName, NoisePerc = Function_Classifier.AddNoise_NPLC(root)########Ask For Adding Noise
        else : 
            NoiseName=0
            NoisePerc=0
        os.chdir(root)
        x_test = list()
        y_test = list()
        for i in range(1,12):####Loading...
            os.chdir(Function_Classifier.Functions_Name(i))
            for j in range(NFiles):#Counting Files in every Folder
                filedata = list()
                if j<10 : P ='-00'
                else : P = '-0'
                FileName = Function_Classifier.Functions_Name(i) + P + str(j) + '.csv'
                print('Loading Test Data...',round((i/11)*100, 2),'%                                          ',end='\r')
                FH = open(FileName)
                Reader = csv.reader(FH)
                filedata = list(Reader)
                for k in range(NDatainFiles):#Counting record in every File
                    filedata[k][0]=float(filedata[k][0])
                    filedata[k][1]=float(filedata[k][1])
                x_test.append(filedata)
                FH.close()
                y_test.append(i-1)
            os.chdir(root)
        xte=np.array(x_test)
        yte=np.array(y_test)
        os.chdir(path)
        shutil.rmtree(root)#Dlete The Test Data Folder
        print('Done.                                                                          ')
        return(xte, yte, NoiseName, NoisePerc)
#######################################################################################################
    def Load_NPLC():
        folders = os.listdir()
        print('\nList of Saved Models of Non Pre Learning Classifier :\n')
        for i in range(len(folders)):
            print(i+1,'-',folders[i],'\n')
        print('\n\nPlease Choose a Model for Loading(0 for back): ',end='')
        Choice = int(input())-1
        os.chdir(folders[Choice])
        ModelFile = folders[Choice]+'.h5'
        Model = models.load_model(ModelFile)
        return(Model, folders[Choice])
#######################################################################################################
    def Functions_Name(FunctionN): #return name of function correspond to Function Number which sent
        if FunctionN==1 : return('x2')
        elif FunctionN==2 : return('Sin')
        elif FunctionN==3 : return('Tan')
        elif FunctionN==4 : return('Sqrt')
        elif FunctionN==5 : return('AbsCos')
        elif FunctionN==6 : return('Floor')
        elif FunctionN==7 : return('Discrete')
        elif FunctionN==8 : return('Rational')
        elif FunctionN==9 : return('AbsLn')
        elif FunctionN==10 : return('x3')
        elif FunctionN==11 : return('log2')
        else : return(0)
#######################################################################################################
    def Functions_Number(FunctionN):#return number of function correspond to Function Name which sent
        if FunctionN=='x2' : return(1)
        elif FunctionN=='Sin' : return(2)
        elif FunctionN=='Tan' : return(3)
        elif FunctionN=='Sqrt' : return(4)
        elif FunctionN=='AbsCos' : return(5)
        elif FunctionN=='Floor' : return(6)
        elif FunctionN=='Discrete' : return(7)
        elif FunctionN=='Rational' : return(8)
        elif FunctionN=='AbsLn' : return(9)
        elif FunctionN=='x3' : return(10)
        elif FunctionN=='log2' : return(11)
        else : return(0)
#######################################################################################################
    def Round_Percentage(Num, Percentage):#Return Rounded Percentage of Given Nummber
        Perc=Num*Percentage/100
        if Num>=0:
            return(int(int(Perc*2+1)/2))
        else:
            return(int(int(Perc*2-1)/2))
#######################################################################################################
    def Data_Ordering(Data):#the porpes of this function is ordering (x,y)s based on xs, Data Should be list
        Length=len(Data)
        for i in range(Length):
            for j in range(i+1,Length):
                if Data[i][0]>Data[j][0]:
                    temp=list()
                    temp=Data[i]
                    Data[i]=Data[j]
                    Data[j]=temp
        return(Data)
#######################################################################################################
    def Creating_Random_Data(DataNumber, ForTest):#Make Data of Every Function by the size of DataNumber
        print('\n')
        for i in range(Function_Classifier.NFunctions):   #Up to Now, We Work With 8 Functions
            print('Making Data for',Function_Classifier.Functions_Name(i+1),'...\t\t',end='\r')
            Function_Classifier.DataFile_Maker(i+1, DataNumber, ForTest)
        print('\n\n Done.')
#######################################################################################################
    def DataFile_Maker(FunctionName, DataNumber, ForTest):#Make Data File, Take Data From DataList_Maker and Save in it
        Function = Function_Classifier.Functions_Name(FunctionName)
        if ForTest : FileName = Function + 'Test.csv'
        else : FileName = Function +  '.csv'
        File_Handeler=open(FileName, 'w', newline='')
        writer=csv.writer(File_Handeler)
        data = Function_Classifier.DataList_Maker(FunctionName, DataNumber)
        writer.writerows(data)
        File_Handeler.close()
#######################################################################################################
    def DataList_Maker(FunctionNumber, DataNumber):#By Given Function Number and Number of Data,it make a list of Data
        data=list()
        for i in range(DataNumber):
            line=list()
            line = Function_Classifier.Make_Pair_Data(FunctionNumber)
            data.append(line)
        data=Function_Classifier.Data_Ordering(data)#ordred
        return(data)
#######################################################################################################
    def Make_Pair_Data(FuncName):#Makeing a pair Data base on User Choises for correct or noisy data
        Pair=list()
        if FuncName==1:
            Pair.append(random.uniform(-5,5))
            Pair.append(math.pow(Pair[0], 2))
            return(Pair)
        elif FuncName==2:
            Pair.append(random.uniform(0, 2*math.pi))
            Pair.append(math.sin(Pair[0]))
            return(Pair)
        elif FuncName==3:
            P = np.random.choice([1,2,3], p=[0.35,0.30,0.35]) #Pi/2=1.57 because of this Function has bias to infinity
            if P==1 : Pair.append(random.uniform(-1.54, -1.4))#I Use 1.54 for keep the answers limited [-30,30]
            elif P==2 :  Pair.append(random.uniform(-1.4, 1.4))
            elif P==3 :  Pair.append(random.uniform(1.4, 1.54))
            Pair.append(math.tan(Pair[0]))
            return(Pair)
        elif FuncName==4:
            Pair.append(random.uniform(0,100))
            Pair.append(math.pow(Pair[0], 0.5))
            return(Pair)
        elif FuncName==5:
            Pair.append(random.uniform(0,2*math.pi))
            Pair.append(abs(math.cos(Pair[0])))
            return(Pair)
        elif FuncName==6:
            Pair.append(random.uniform(-2,2))
            Pair.append(math.floor(Pair[0]))
            return(Pair)
        elif FuncName==7:
            Pair.append(random.uniform(-5,10))
            if Pair[0]<0:
                Pair.append(Pair[0]*Pair[0]-2)
            else:
                Pair.append(math.sqrt(Pair[0]))
            return(Pair)
        elif FuncName==8:
            y=10 #Check Point
            P = np.random.choice([1,2], p=[0.6,0.4])#Non Uniform Selectioin for better Learning
            while (y>5 or y<-5):
                if P==1 :    
                    x = random.uniform(-8,8)
                elif P==2:
                    PP = np.random.choice([1,2,3,4], p=[0.25,0.25,0.25,0.25])
                    if PP==1 : x = random.uniform(-3,-2.13)#We need more data in this interval
                    elif PP==2 : x = random.uniform(-1.89,-1.52)#We need more data in this interval
                    elif PP==3 : x = random.uniform(1.52,1.89)#We need more data in this interval
                    elif PP==4 : x = random.uniform(2.13,3)#We need more data in this interval
                y=(math.pow(x,2)+1)/(math.pow(x,2)-4)
            Pair.append(x)
            Pair.append(y)
            return(Pair)
        elif FuncName==9:
            Pair.append(random.uniform(0,10))
            Pair.append(abs(np.log(Pair[0])))
            return(Pair)
        elif FuncName==10:
            P = np.random.choice([1,2,3], p=[0.35,0.30,0.35]) 
            if P==1 : Pair.append(random.uniform(-3, -1.5))
            elif P==2 :  Pair.append(random.uniform(-1.5, 1.5))
            elif P==3 :  Pair.append(random.uniform(1.5, 3))
            Pair.append(math.pow(Pair[0],3))
            return(Pair)
        elif FuncName==11:
            Pair.append(random.uniform(0,10))
            Pair.append(np.log2(Pair[0]))
            return(Pair)
        return(0)
#######################################################################################################
    def New_Project_PreLearn():# Start a New Project for Pre Learning Classifier
        print('Please Enter New Project''s Name (0 for back): ',end='')
        Name = input()
        if Name=='0' : return()
        os.mkdir(Name)
        os.chdir(Name)
        Path=os.getcwd()
        DataN = int(input('Please Enter Data Number You Want to Creat : '))
        Function_Classifier.Creating_Random_Data(DataN, False)#Creating Training Data
        NoiseName, Percentage = Function_Classifier.AddNoise_PLC(Path)#Ask For Add noise to Data
        x_train ,y_train ,x_test, y_test, NxTrain, NxTest, FNames = Function_Classifier.Data_Loader(Path)#Load Data
        Iteration=int(input('\n\nPlease Enter the Iteration of Neural Networks Learnings: '))           
        for i in range(Function_Classifier.NFunctions):#Run NN For Every Function,Build a Model and Save it
            Model = Function_Classifier.RunNN_ToLearnFunctions(Iteration, x_train[i], y_train[i], x_test[i], y_test[i], NxTrain[i], NxTest[i], FNames[i])
            ModelFileName = FNames[i].replace('csv','h5')
            Model.save(ModelFileName)
        TestDataNumber = Function_Classifier.Round_Percentage(DataN, 25)
        Function_Classifier.Creating_Random_Data(TestDataNumber, True)#Creating Test Data
        ProjectInfo=list()#Saving Project Information
        ProjectInfo.append(DataN)
        ProjectInfo.append(Iteration)
        ProjectInfo.append(NoiseName)
        ProjectInfo.append(Percentage)
        ProjectInfoFileName=Name + '.csv'
        FileHandeler=open(ProjectInfoFileName,'w',newline='')
        Writer=csv.writer(FileHandeler)
        Writer.writerow(ProjectInfo)
        FileHandeler.close()
        Function_Classifier.ActiveProject=False
########################################################################################################
    def AddNoise_PLC(Path):###Adding Noise To a Project of PLC 
        Perc=0
        Noise_Name=0
        ans=input('\n\nDo You Want add noise To Data(y/n)? ')
        if ans!='y' and ans!='Y' : return(Noise_Name, Perc)
        os.chdir(Path)
        Dirlist = os.listdir()
        print('\nHere are Noises You Can Choose For Adding to Data Sets:\n\n1- Gaussian Noise\n2- Rayleigh Noise'
              '\n3- Gamma Noise\n4- Exponential Noise\n5- Uniform Noise\n\nPlease Enter The Number of The Noise You Want '
              'Add To Data Sets: ',end='')
        Noise_Name=int(input())
        if 0<Noise_Name<6:
            print('Please Enter The Percentage of Noise You Want to Add(1-100) : ',end='')
            Perc=int(input())
            for i in range(Function_Classifier.NFunctions):
                FileHandeler = open(Dirlist[i])
                Reader = csv.reader(FileHandeler)
                data = list(Reader)
                FileHandeler.close()
                for j in range(len(data)):
                    data[j][0]=float(data[j][0])
                    data[j][1]=float(data[j][1])
                data = Noise_Management.Add_Noise(data, Noise_Name, Perc)
                FileHandeler = open(Dirlist[i], 'w', newline='')
                Writer=csv.writer(FileHandeler)
                Writer.writerows(data)
                FileHandeler.close()
        return(Noise_Name, Perc)
#######################################################################################################
    def DataSpliter(Data, Percentage):
        x_test=list()
        x_train=list()
        y_test=list()
        y_train=list()
        Length = len(Data)
        NTest = Function_Classifier.Round_Percentage(Length, Percentage)
        NTrain = Length - NTest
        for i in range(Length):
            if i<NTrain-1:
                x_train.append(Data[i][0])
                y_train.append(Data[i][1])
            else : 
                x_test.append(Data[i][0])
                y_test.append(Data[i][1])
        return(x_train, y_train, x_test, y_test, NTrain, NTest)
#######################################################################################################
    def Data_Loader(path):
        x_train=list()
        y_train=list()
        x_test=list()
        y_test=list()
        NxTrain=list()
        NxTest=list()
        FunctionName=list()
        os.chdir(path)
        DirList = os.listdir()
        print('\n')
        for i in range(len(DirList)):
            print('Loading Data...',((i+1)/len(DirList))*100,'%',end='\r')
            FunctionNumber = Function_Classifier.Functions_Number(DirList[i])
            data=list()
            FunctionName.append(DirList[i])
            FileName = DirList[i]
            FileHandeler = open(FileName)
            ReadFile = csv.reader(FileHandeler)
            data=list(ReadFile)
            for k in range(len(data)):#Change Data To float
                data[k][0]=float(data[k][0])
                data[k][1]=float(data[k][1])
            x_train_i, y_train_i, x_test_i, y_test_i, NxTrain_i, NxTest_i = Function_Classifier.DataSpliter(data, 25)
            NxTrain.append(NxTrain_i)
            NxTest.append(NxTest_i)
            x_train.append(x_train_i)
            y_train.append(y_train_i)
            x_test.append(x_test_i)
            y_test.append(y_test_i)
            FileHandeler.close()
            print('                                                                  ',end='\r')
        print('\nLoading Data is Completed...')
        X=np.array(x_train)                
        Y=np.asarray(y_train)
        x=np.array(x_test)
        y=np.asarray(y_test)
        return(X, Y, x, y, NxTrain, NxTest, FunctionName)
#######################################################################################################
    def RunNN_ToLearnFunctions(Iteration, x_train, y_train, x_test, y_test, NxTrain, NxTest, FName):#Comes From Load_Project_Data
        print('\n\nLearning',FName.replace('.csv',''),':')
        network = models.Sequential()
        #Add fully connected layer with a Tanh activation function
        network.add(layers.Dense(units=64, activation="tanh", input_shape=(1,)))
        network.add(layers.Dense(units=32, activation="tanh"))
        network.add(layers.Dense(units=32, activation="tanh"))
        network.add(layers.Dense(units=16, activation="tanh"))
        network.add(layers.Dense(units=1))
        network.compile(loss="mse", #Mean squared error
                        optimizer="adam", #optimization algorithm
                        metrics=["accuracy"])
        for i in range(Iteration):
            print('epochs #: ',i+1,'/',Iteration, end='')
            history = network.fit(x_train, #features
                                  y_train, #Target vector
                                  epochs=1, #number of echos
                                  verbose=0,# No output
                                  batch_size=1, #number of observation per batch
                                )#Test data
            scores = network.evaluate(x_train, y_train, verbose=0)
            print('\tScore is : ',scores,'\t\t',end='\r')
        return(network)
########################################################################################################
    def Load_Project():
        ProjectList = os.listdir()
        if len(ProjectList)==0:
            print('\nThere is No Projects...\nPress Enter To Continue...',end='')
            input()
            return(0,0,0,0,0)
        for i in range(len(ProjectList)):
            print('\n',i+1,'- ',ProjectList[i])
        Choice=int(input('\nPlease Choose a Project : '))
        if 0<Choice<len(ProjectList)+1:###################################################################################################################Main Part##############
            ProjectModels=list()#Keep Models
            ModelsTestData=list()
            ModelsX_Test=list()
            ModelsY_Test=list()
            ProjectName=ProjectList[Choice-1]
            os.chdir(ProjectName)
            Path=os.getcwd()
            ProjectInfoFileName=ProjectName+'.csv'
            ProjectFileH = open(ProjectInfoFileName)
            Reader=csv.reader(ProjectFileH)
            ProjectInfo=list(Reader)
            ProjectFileH.close()
            print('\n')
            for i in range(Function_Classifier.NFunctions):
                print('Loading Models...',end='')
                ModelFileName=Function_Classifier.Functions_Name(i+1)+'.h5'
                ModelTestDataF=Function_Classifier.Functions_Name(i+1)+'Test.csv'
                FHandeler=open(ModelTestDataF)
                Reader=csv.reader(FHandeler)
                data=list(Reader)
                FHandeler.close()
                x_test=list()
                y_test=list()
                for j in range(len(data)):
                    data[j][0]=float(data[j][0])
                    data[j][1]=float(data[j][1])
                    x_test.append(data[j][0])
                    y_test.append(data[j][1])
                ProjectModels.append(models.load_model(ModelFileName))
                ModelsTestData.append(data)
                ModelsX_Test.append(x_test)
                ModelsY_Test.append(y_test)
                print(round((i/Function_Classifier.NFunctions)*100,2),'%\t\t',end='\r')
            ProjectModelsScores=list()
            for i in range(Function_Classifier.NFunctions):#Show The Models Summary and Scores
                #ProjectModels[i].summary()
                print('This Model Trained for ',Function_Classifier.Functions_Name(i+1),'with',ProjectInfo[0][0],'in',ProjectInfo[0][1],'Iteration')
                if int(ProjectInfo[0][2]==0) : print('Model Data Has Not Any Noise.')
                else : print('The Model Data has', ProjectInfo[0][3], '%',' of with',Noise_Management.Noise_Name(int(ProjectInfo[0][2])),'Noise.')
                ProjectModelsScores.append(ProjectModels[i].evaluate(ModelsX_Test[i],ModelsY_Test[i],verbose=0))
                print('\nAnd The Model Score is : ',ProjectModelsScores[i])
            Function_Classifier.ActiveProject=True
            return(ProjectInfo, ProjectModels, ModelsX_Test, ModelsY_Test, 1)
        else: 
            print('Wrong Choice...')
            return(0,0,0,0,-1)
########################################################################################################
    def Make_Classifier_TestData(classifier_info):
        noise=False
        random.seed(1)
        Functions = list()
        Data = list()
        X_test = list()
        Y_test = list()
        FileHandeler = open('Classifier_TestData.csv','w',newline='')
        ans=input('\n\nDo You Want add noise To Test DataSet(y/n)? ')
        if ans=='y' or ans=='Y' : 
            print('Here are Noises You Can Choose For Adding to Data Sets:\n\n1- Gaussian Noise\n2- Rayleigh Noise'
                  '\n3- Gamma Noise\n4- Exponential Noise\n5- Uniform Noise\n\nPlease Enter The Number of The Noise You Want '
                  'Add To Data Sets: ',end='')
            Noise_Name=int(input())
            if 0<Noise_Name<6:
                print('Please Enter The Percentage of Noise You Want to Add(1-100) : ',end='')
                Perc=int(input())
                noise=True
        for i in range(110):
            FunctionNum = random.randint(1,11)
            Functions.append(Function_Classifier.Functions_Name(FunctionNum))
            Fdata=list()
            x_test=list()
            y_test=list()
            for j in range(Function_Classifier.Round_Percentage(int(classifier_info[0][0]),25)):
                Fdata.append(Function_Classifier.Make_Pair_Data(FunctionNum))
            if noise : Fdata = Noise_Management.Add_Noise(Fdata, Noise_Name, Perc)
            for j in range(Function_Classifier.Round_Percentage(int(classifier_info[0][0]),25)):
                x_test.append(Fdata[j][0])
                y_test.append(Fdata[j][1])
            Data.append(Fdata)
            X_test.append(x_test)
            Y_test.append(y_test)
        Writer = csv.writer(FileHandeler)
        Writer.writerows(Data)
        FileHandeler.close()
        return(Data, X_test, Y_test, Functions)
########################################################################################################
    def Decision_Maker(ClassifierInfo, Models, X_test, Y_test):
        Prediction=list()
        for i in range(len(X_test)):
            scores=list()
            for j in range(Function_Classifier.NFunctions):
                scores.append(Models[j].evaluate(X_test[i], Y_test[i], verbose=0))
            Prediction.append(scores.index(min(scores))+1)
        return(Prediction)        
########################################################################################################
    def PreLearning_Classifier():
        ClassifierInfo , ClassifierModels, ModelsXs, ModelsYs, LoadingStatus = Function_Classifier.Load_Project()
        if LoadingStatus==1 : 
            for i in range(Function_Classifier.NFunctions):
                print(i+1,'-',Function_Classifier.Functions_Name(i+1),'Model with Score = '
                      , ClassifierModels[i].evaluate(ModelsXs[i], ModelsYs[i],verbose=0))
                print('The Models have',ClassifierInfo[0][3],'%','of',Noise_Management.Noise_Name(int(ClassifierInfo[0][2])),'\n')
        else : return(0)
        TestData, X_test, Y_test, TestF = Function_Classifier.Make_Classifier_TestData(ClassifierInfo)
        Predictions = Function_Classifier.Decision_Maker(ClassifierInfo, ClassifierModels, X_test, Y_test)
        ClassifierScore = 0
        for i in range(len(Predictions)):
            print('\nClassifier Prediction on',i+1,'th Test Data is: ',Function_Classifier.Functions_Name(Predictions[i]),
                   '\t\tand The True Function is :',TestF[i],end='')
            if Function_Classifier.Functions_Number(TestF[i])==Predictions[i] : ClassifierScore = ClassifierScore + 1
            else : print('\t Wrong!!',end='')
        print('\n\nThe Score of Classifier is : ',(ClassifierScore/len(TestData))*100,'\n\nThe Classifier is predicit',ClassifierScore,' out of ',len(TestData)
               ,'Test Data.\n\nPress Enter To Continue...')    
        input()            
########################################################################################################
    def Loading_Constant_Data_For_NPLC(path):
        os.chdir(path)
        folders = os.listdir()
        if len(folders)==0 : return(0)
        print('\nList of Saved Data for Non Pre Learning Classifier :\n')
        for i in range(len(folders)):
            print(i+1,'-',folders[i],'\n')
        print('\n\nPlease Choose a Data Folder for Loading : ',end='')
        Choice = int(input())-1
        os.chdir(folders[Choice])
        dataf = folders[Choice] + '.csv'
        FH = open(dataf)
        Reader = csv.reader(FH)
        filedata = list(Reader)
        NFD = int(filedata[0][0])
        NPD = int(filedata[0][1])
        DataPath = os.getcwd()
        x_train, y_train, x_test, y_test = Function_Classifier.LoadNetData_4_NPLC(DataPath,NFD, NPD)
        return(x_train, y_train, x_test, y_test, NFD, NPD)
########################################################################################################
    def Constant_Data_addNoise(Data, NFD):
        Noise_Name = 0
        Perc = 0
        print('\nHere are Noises You Can Choose For Adding to Data Sets:\n\n1- Gaussian Noise\n2- Rayleigh Noise'
              '\n3- Gamma Noise\n4- Exponential Noise\n5- Uniform Noise\n\nPlease Enter The Number of The Noise You Want '
              'Add To Data Sets: ',end='') 
        Noise_Name=int(input())
        if 0<Noise_Name<6:
            print('Please Enter The Percentage of Noise You Want to Add(1-100) : ',end='')
            Perc=int(input())
        else: return(Noise_Name, Perc)
        for i in range(Function_Classifier.NFunctions):
            for j in range(NFD):
                Data[i][j] = Noise_Management.Add_Noise(Data[i][j], Noise_Name, Perc)
        return(Data, Noise_Name, Perc)
########################################################################################################
    def Models_by_CData(path):
        NoiseName = 0
        Perc = 0
        x_train, y_train, x_test, y_test, NFD, NPD = Function_Classifier.Loading_Constant_Data_For_NPLC(path)
        ans = input('\nDo You Want to Add Noise To Data(y/n)? ')
        if ans=='y' or ans=='Y' : 
            x_train, NoiseName, Perc = Function_Classifier.Constant_Data_addNoise(x_train, NFD)
        ans = input('\nDo You Want to Add Noise To Test Data(y/n)? ')
        if ans=='y' or ans=='Y' : Function_Classifier.Constant_Data_addNoise(x_test, Function_Classifier.Round_Percentage(NFD,25))
        Iteration = int(input('\nPlease Enter Number of Iteration : '))
        Model = Function_Classifier.RunNNForNPLC(x_train, y_train, x_test, y_test, NFD, NPD, Iteration)
        ModelsPath = path.replace('Constant', 'ConsModels')
        ModelName = input('\n\nEnter a Name for Model : ')
        os.chdir(ModelsPath)
        os.mkdir(ModelName)
        os.chdir(ModelName)
        ModelInfo=list()
        ModelInfo.append(NFD)#0  Number of Files of Data
        ModelInfo.append(NPD)#1 Number of Data Point in every File
        ModelInfo.append(Iteration)#2 Number of Iteration
        ModelInfo.append(NoiseName)#3 Noise Name
        ModelInfo.append(Perc)#4 Noise Percentage 0=No Noise
        if Function_Classifier.DropOut: ModelInfo.append(1)#5 Drop Out Layer 1 = DropOut Layer On
        else : ModelInfo.append(0)
        ModelInfo.append(Function_Classifier.DropOut_Rate)#6 Rate Of Drop Out Layer
        ModelFileName = ModelName + '.h5'
        ModelInfoFile = ModelName + '.csv'
        Model.save(ModelFileName)
        FH = open(ModelInfoFile, 'w' , newline='')
        Writer = csv.writer(FH)
        Writer.writerow(ModelInfo)
        FH.close()
########################################################################################################
    def Use_ConsModel(path) : 
        os.chdir(path)
        folders = os.listdir()
        print('\nList of Saved Models :\n')
        for i in range(len(folders)):
            print(i+1,'-',folders[i],'\n')
        print('\n\nPlease Choose a Model for Loading(0 for back): ',end='')
        Choice = int(input())-1
        os.chdir(folders[Choice])
        ModelFile = folders[Choice]+'.h5'
        ModelInfoFile = folders[Choice]+'.csv'
        Model = models.load_model(ModelFile)
        ModelInfo=list()
        FH = open(ModelInfoFile)
        Reader = csv.reader(FH)
        ModelInfo = list(Reader)
        for i in range(len(ModelInfo[0])-1): ModelInfo[0][i]=int(ModelInfo[0][i])
        ModelInfo[0][len(ModelInfo[0])-1] = float(ModelInfo[0][len(ModelInfo[0])-1])
        Model.summary()#####################################################
        print('The Model was Trained by',ModelInfo[0][0],'Data Files for every Function',
               '\nwich every File Contains',ModelInfo[0][1],'Data.',
               '\nThe Model Trained in',ModelInfo[0][2],'Iteration.',end="")
        if not ModelInfo[0][3]==0 :
            print('\nThe Model Data has',ModelInfo[0][4],'% Of', Noise_Management.Noise_Name(ModelInfo[0][3]))
        else : print('The Model Data Has not Any Noise.')    
        if (ModelInfo[0][5]==1) : print('The Model has Drop Out Layer with the Rate of ',ModelInfo[0][6])
        else : print('The Model Has no Drop Out Layer.')
        NumData = int(input('\nPlease Enter The Number of Test Data for Evalution of Model : '))
        x_test , y_test, NoiseName, NoisePerc = Function_Classifier.Making_TestDataSet_NPLC(NumData,ModelInfo[0][1], True)
        testprediction = Model.predict([x_test],batch_size=1)
        Scores = Model.evaluate(x_test, y_test, verbose=0)
        Success=0
        ch = input('Do You Want to see Every Test Result(y/n)? ')
        if ch=='y' or ch=='Y' : show=True
        else : show=False
        for i in range(len(y_test)):
            Max , Index = Function_Classifier._MaxOfRow(testprediction[i])
            if (show) : print('Model Prediction is :',Function_Classifier.Functions_Name(Index+1),'with',round(Max*100,2),'%','and the true answer is :',Function_Classifier.Functions_Name(y_test[i]+1)) 
            if (Index==y_test[i]):
                Success+=1
        print('\n\nFrom ',len(y_test),' Test Data, Our Model\'s Correct Prediction was',Success,
               '\nand Success Percentage is :',round((Success/len(y_test))*100,2),'%')
        if NoiseName==0 : print('\nThese Test Dataset Have no Noise.')
        else : print('These Test Dataset Have',NoisePerc,'% Noise of ',
        Noise_Management.Noise_Name(NoiseName),'Noise.')
        print('\nThe Score of Model is : ',Scores,'\n\nPress Enter To Continue...',end='')
        input()
########################################################################################################
    def Making_CData_For_NPLC(path):#Making Data For Constant Using in NPLC
        os.chdir(path)
        DataName = input('\nPlease Enter the Data Name : ')
        DataFileNum = int(input('\nPlease Enter The Number of Files You Want Creat for Every Function : '))
        DataPointNum = int(input('\nPlease Enter Number of Point You Want Creat in Files : '))
        os.mkdir(DataName)
        os.chdir(DataName)
        root = os.getcwd()
        for i in range(1,12):
            os.mkdir(Function_Classifier.Functions_Name(i))
            os.chdir(Function_Classifier.Functions_Name(i))
            for j in range(DataFileNum):
                if j<10  : P = '-00' 
                else : P = '-0'
                FileName = Function_Classifier.Functions_Name(i) + P + str(j) + '.csv'
                print('Creating Files of ',Function_Classifier.Functions_Name(i),': ',FileName,'           ',end='\r')
                FunctionList = Function_Classifier.DataList_Maker(i, DataPointNum)
                FH = open(FileName, 'w', newline='')
                writer = csv.writer(FH)
                writer.writerows(FunctionList)
                FH.close()
            os.chdir(root)
        os.chdir(root)
        data=list()
        data.append(DataFileNum)
        data.append(DataPointNum)
        FileName = DataName + '.csv'
        FH = open(FileName, 'w', newline='')
        writer = csv.writer(FH)
        writer.writerow(data)
        FH.close()
########################################################################################################
########################################################################################################     