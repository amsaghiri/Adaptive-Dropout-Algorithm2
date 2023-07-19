#در این برنامه به انختاب کاربر یک تابع با ورودی‌های انتخابی کاربر برای یادگیری به شبکه عصبی داده 
#می‌شود و مراحل یادگیری و آموزش و تست به نمایش گذاشته می‌شود
# Load liabraries
import math 
import csv
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from sklearn import preprocessing
from Data_Manager import DataSets_Maker
#############################################################
#با ورودی‌های داده‌های آزمایشی و آموزشی و تعداد چرخه آموزش یک شبکه عصبی را ایجاد و اجرا می‌کند
#با استفاده از داده‌ها آن‌ها را بر روی نمودار نمایش می‌دهد
#Title & ytitl صرفاً برای نمایش روی نمودار استفاده می‌شوند
class Neural_Net():
    Model_Directory=''#Model Directory Path
    Model_Name=''#Model Name Same as Directory Name
    Model_TrainSetFilename=''
    Function_N=0#int
    Noise_Function_N=0#int
    Noise_Percentage=0#int
    N_TrainigSet=0#Number of Training Data Set
    N_Iteration=0#Number Of NN Teration
    ModelF=''#Name of Function Used For Make The Model
    NoiseFN=''#Name of Noise that used in Model
#########################################################
    def Model_Data_List(ModelData):
        ModelData.append(Neural_Net.Function_N)
        ModelData.append(Neural_Net.Noise_Function_N)
        ModelData.append(Neural_Net.Noise_Percentage)
        ModelData.append(Neural_Net.N_TrainigSet)
        ModelData.append(Neural_Net.N_Iteration)
        ModelData.append(Neural_Net.Model_TrainSetFilename)
        return(ModelData)
#########################################################
    def Model_DataSetFileNames():
        if Neural_Net.Function_N==1: 
            Neural_Net.Model_TrainSetFilename='x2TrainSet.csv'
        elif Neural_Net.Function_N==2:
            Neural_Net.Model_TrainSetFilename='SinTrainSet.csv'
        elif Neural_Net.Function_N==3: 
            Neural_Net.Model_TrainSetFilename='TanTrainSet.csv'
        elif Neural_Net.Function_N==4: 
            Neural_Net.Model_TrainSetFilename='SqrtTrainSet.csv'
        elif Neural_Net.Function_N==5: 
            Neural_Net.Model_TrainSetFilename='AbsCosTrainSet.csv'
        elif Neural_Net.Function_N==6:
            Neural_Net.Model_TrainSetFilename='FloorTrainSet.csv'
        elif Neural_Net.Function_N==7:
            Neural_Net.Model_TrainSetFilename='DiscreteTrainSet.csv'
        elif Neural_Net.Function_N==8:
            Neural_Net.Model_TrainSetFilename='RationalTrainSet.csv'
        elif Neural_Net.Function_N==9:
            Neural_Net.Model_TrainSetFilename='AbsLnTrainSet.csv'
        elif Neural_Net.Function_N==10:
            Neural_Net.Model_TrainSetFilename='x3TrainSet.csv'
        elif Neural_Net.Function_N==11:
            Neural_Net.Model_TrainSetFilename='LogTrainSet.csv'
#########################################################
    def Nameing():
        if Neural_Net.Function_N==1 : Neural_Net.ModelF='X^2'
        elif Neural_Net.Function_N==2 : Neural_Net.ModelF='Sin(x)'
        elif Neural_Net.Function_N==3 : Neural_Net.ModelF='Tan(x)'
        elif Neural_Net.Function_N==4 : Neural_Net.ModelF='Sqrt(X)'
        elif Neural_Net.Function_N==5 : Neural_Net.ModelF='|Cos(x)|'
        elif Neural_Net.Function_N==6 : Neural_Net.ModelF='[x]'
        elif Neural_Net.Function_N==7 : Neural_Net.ModelF='Discrete --> F(x) = Sqrt(x) & x^2-2'
        elif Neural_Net.Function_N==8 : Neural_Net.ModelF='(x^2+1)/(x^2-5)'
        elif Neural_Net.Function_N==9 : Neural_Net.ModelF='|Ln(x)|'
        elif Neural_Net.Function_N==10 : Neural_Net.ModelF='X^3'
        elif Neural_Net.Function_N==11 : Neural_Net.ModelF='log(x)'
        if Neural_Net.Noise_Percentage>0 : 
            if Neural_Net.Noise_Function_N==1 : Neural_Net.NoiseFN='Gaussian Noise'
            elif Neural_Net.Noise_Function_N==2 : Neural_Net.NoiseFN='Rayleigh Noise'
            elif Neural_Net.Noise_Function_N==3 : Neural_Net.NoiseFN='Gamma Noise'
            elif Neural_Net.Noise_Function_N==4 : Neural_Net.NoiseFN='Exponential Noise'
            elif Neural_Net.Noise_Function_N==5 : Neural_Net.NoiseFN='Uniform Noise'
        else : Neural_Net.NoiseFN='Whitout Noise'
#########################################################
    def LoadModel_TestDataMaker(NewData):   
        if NewData :
            testfile=Neural_Net.Model_Name.replace('.h5','testset.csv')
            File_Handeler=open(testfile, "w", newline="")
            W=csv.writer(File_Handeler)
            TestN = DataSets_Maker.Round_Percentage(Neural_Net.N_TrainigSet, 25)
            for i in range(TestN):
                TestPair=list()
                TestPair = DataSets_Maker.Make_Pair_Data(Neural_Net.Function_N)
                W.writerow(TestPair)
            File_Handeler.close()
        else:
            Neural_Net.Model_DataSetFileNames()
            testfile=Neural_Net.Model_TrainSetFilename.replace('Train','Test')
        FHandeler=open(testfile)
        readfile=csv.reader(FHandeler)
        data=list(readfile)
        NxTe=len(data)
        x_te = np.zeros((NxTe,1))
        y_te = np.zeros((NxTe,1))
        for i in range(NxTe):#Change Data Type For Sorting
            data[i][0]=float(data[i][0])
            data[i][1]=float(data[i][1])
        for line in range(NxTe):
            x_te[line]=data[line][0] 
            y_te[line]=data[line][1]
        FHandeler.close()
        return(x_te, y_te)
#########################################################
    def Load_Network(): # Load a Saved network from any Place
        files=os.listdir()
        print('Here is The Saved Models: \n')
        for i in range(len(files)) : 
            print(i+1,'-',files[i],'\n')
        print('Choose a Model for Loading : ',end='')
        Choice=int(input())-1
        if Choice not in range (len(files)) : 
            print('Error - Wrong Choice...')
            return(0)
        print('\n',files[Choice],'Model is Loading, Please Wait...', end='')
        os.chdir(files[Choice])#get into Model Directory
        Neural_Net.Model_Directory=os.getcwd()
        Neural_Net.Model_Name=files[Choice]+'.h5'
        Model=models.load_model(Neural_Net.Model_Name)######>>>Loading Model
        ModelDataFile=Neural_Net.Model_Name.replace('.h5','.csv')
        FHandeler=open(ModelDataFile)
        readfile=csv.reader(FHandeler)
        ModelD=list(readfile)#Load Data Needed To Run Network
        Neural_Net.Function_N=int(ModelD[0][0])
        Neural_Net.Noise_Function_N=int(ModelD[0][1])
        Neural_Net.Noise_Percentage=int(ModelD[0][2])
        Neural_Net.N_TrainigSet=int(ModelD[0][3])
        Neural_Net.N_Iteration=int(ModelD[0][4])
        Neural_Net.Model_TrainSetFilename=ModelD[0][5]
        Neural_Net.Nameing()
        return(Model)
#########################################################
    def Save_Network(Model):#Save the Trained Model in Models Directory
        ModelDataFilename=Neural_Net.Model_Name+'.csv'######Saving Model Training Data
        FileHandeler=open(ModelDataFilename,'w',newline='')
        Writer=csv.writer(FileHandeler)
        ModelData=list()
        ModelData=Neural_Net.Model_Data_List(ModelData)
        Writer.writerow(ModelData)
        FileHandeler.close()
        ModelFilename=Neural_Net.Model_Name+'.h5'########Save The Model ItSelf
        Model.save(ModelFilename)
#########################################################
    def Run_Neural_Network(Iteration, x_train, y_train, x_test, y_test, NxTrain, NxTest, Title, ytitle, Save):
        print("Starting Neural Network...")
        #start neural network
        network = models.Sequential() 
        network.add(layers.Dense(units=64, activation="tanh",  input_shape=(1,)))###Layer1
        network.add(layers.Dense(units=32, activation="tanh",))
        network.add(layers.Dense(units=32, activation="tanh"))
        network.add(layers.Dense(units=16, activation="tanh"))
        network.add(layers.Dense(units=1))###Layer 5
        #compile neural network
        network.compile(loss="mse", #Mean squared error
                        optimizer="adam", #optimization algorithm
                        metrics=["accuracy"])
        #Train neural network
        max=0
        min=100
        for i in range(Iteration):
            plt.cla()
            print("epochs #: ",i,'/',Iteration,end='')
            history = network.fit(x_train, #features
                                  y_train, #Target vector
                                  epochs=1, #number of echos
                                  verbose=0,# No output
                                  batch_size=1, #number of observation per batch
                                )#Test data
            prediction = network.predict([x_train],batch_size=NxTrain)#plotting after every train
            scores = network.evaluate(x_train, y_train, verbose=0)
            print('\tScore is : ',scores)
            if scores[0]>max : max=scores[0]
            if scores[0]<min : min=scores[0]
            plt.title(Title)
            plt.ylabel(ytitle)
            plt.plot(x_train,y_train,'b.',label='Real Data')
            plt.plot(x_train,prediction,'g.',scaley=False,label='Prediction')
            plt.legend(loc='upper right', fontsize='small')
            plt.draw()
            plt.pause(0.000001)
            #plotting test set    
        print("\n\nScores Max is : ",max,'\t\tScores Min is : ', min)
        if (Save) : Neural_Net.Save_Network(network) #Save the Trained Model
        Neural_Net.Use_Model(network,x_test,y_test)
        return(prediction)
#########################################################
    #برای بارگذاری داده‌های آموزشی و آزمایشی از فایل‌های دلخواه 
    def Load_Network_Data(filename, testfilename):
        FHandeler=open(filename)
        readfile=csv.reader(FHandeler)
        data=list(readfile)
        NxTr=len(data)
        x_tr = np.zeros((NxTr,1))
        y_tr = np.zeros((NxTr,1))
        for i in range(NxTr):#change Data Type For Sorting
            data[i][0]=float(data[i][0])
            data[i][1]=float(data[i][1])
        #data.sort()#Sort For Plot
        for line in range(NxTr):
            x_tr[line]=data[line][0] 
            y_tr[line]=data[line][1]
        #بارگذاری مجموعه داده‌های آزمایشی
        FHandeler=open(testfilename)
        readfile=csv.reader(FHandeler)
        data=list(readfile)
        NxTe=len(data)
        x_te = np.zeros((NxTe,1))
        y_te = np.zeros((NxTe,1))
        for i in range(NxTe):#Change Data Type For Sorting
            data[i][0]=float(data[i][0])
            data[i][1]=float(data[i][1])
         #data.sort()#Sort For Plot
        for line in range(NxTe):
            x_te[line]=data[line][0] 
            y_te[line]=data[line][1]
        return(NxTr, x_tr, y_tr, NxTe, x_te, y_te)
#########################################################
    def Use_Model(Model,X_Test, Y_Test):
        Model.summary()
        Neural_Net.Nameing()
        print('The Model was Trained For f(x) = ',Neural_Net.ModelF,end="")
        if Neural_Net.Noise_Percentage>0 : print(' With ',Neural_Net.Noise_Percentage,'%','of',Neural_Net.NoiseFN)
        else : print('',Neural_Net.NoiseFN)
        print('This Model was Trained with',Neural_Net.N_TrainigSet,'train data and in',Neural_Net.N_Iteration,'Iterations.')
        score=Model.evaluate(X_Test,Y_Test,verbose=0)
        print('The Score of Model is : ',score,'\n\nPress Enter To Continue...',end='')
        input()
        print('\n')
        N_Xtest=len(X_Test)
        prediction=Model.predict([X_Test],batch_size=N_Xtest)
        Loss=0
        for i in range(N_Xtest):
            print('x=',X_Test[i],'\tf(x)=',Neural_Net.ModelF,'=',Y_Test[i],'\tPrediction of Model is: ',prediction[i],'\tAnd the Loss Is: ',math.pow(Y_Test[i]-prediction[i],2))
            Loss=Loss+math.pow(Y_Test[i]-prediction[i],2)
        print('\nTotal Loos is : ', Loss/N_Xtest,'\n\nPress Enter To Continue...',end='')
        ######################
        plt.close() 
        #plt.title('Predicting on Test Data Set with ' + str(N_Xtest) + ' Data Test')
        #plt.ylabel(Neural_Net.ModelF)
        #plt.plot(X_Test,Y_Test,'b.', label='True Data')
        #plt.plot(X_Test,prediction,'g.', label='Predicted on Test Set')
        #plt.legend(loc='upper right', fontsize='small')
        #plt.show()
        ######################
        input()
#########################################################
    #نتایج حاصل را به فایل مشخص اضافه می‌کند
    def WriteTreat_File(Prediction):
        Read_File=open('TreatResult.csv', 'r', newline='')
        reader=csv.reader(Read_File)
        OldData=list(reader)
        Read_File.close()
        NData=len(OldData)
        for i in range(NData):
            OldData[i].append(float(Prediction[i]))
            OldData[i].append(float(OldData[i][0])-float(Prediction[i]))
        Write_File=open('TreatResult.csv', 'w', newline='')
        Writer=csv.writer(Write_File)
        Writer.writerows(OldData)
        Write_File.close()
######################################################################################