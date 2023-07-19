# Load liabraries
import math 
import os
import csv
import random
import numpy as np
from Classification_Scenario import Function_Classifier
from Data_Manager import DataSet_Treatment as Treatment
from Data_Manager import DataSets_Maker as DSM
from Data_Manager import Noise_Management as NM
from FiveL_DropOut_Neural_Network import Neural_Net as NeuralNet
import matplotlib.pyplot as plt
import shutil
#############################################################
class Menu():
    Active_Model=False
    Model_Noise=False
    DropOut=False
    DropOut_Rate=0
    root=os.getcwd()
    Models=list()
    NaFuncFoldModels='FunctionsLearningModels'
    NaPreLearModels='PreLearningModels'
    NaNonPreLearModels='NonPreLearningModels'
    Constant = 'Constant'
    ConsModel = 'ConsModels'
#############################################################
    def For_Start():
        Function_Classifier.RootPath=Menu.root
        os.chdir(Menu.root)
        if os.path.exists('models')==False:
            os.mkdir('Models')
            os.chdir('Models')
        else: os.chdir('Models')
        if not os.path.exists(Menu.NaNonPreLearModels) : os.mkdir(Menu.NaNonPreLearModels)           
        if not os.path.exists(Menu.NaPreLearModels) : os.mkdir(Menu.NaPreLearModels)
        if not os.path.exists(Menu.NaFuncFoldModels) : os.mkdir(Menu.NaFuncFoldModels)
        if not os.path.exists(Menu.Constant) : os.mkdir(Menu.Constant)
        if not os.path.exists(Menu.ConsModel) : os.mkdir(Menu.ConsModel)
#############################################################
    def Main_Menu():
        loop=True
        while (loop):
            Menu.For_Start()
            os.system('cls')
            print('\n\t\t\t<<<<< Main Menu >>>>>')
            print('\n\n1- Start a New Model.\n2- Load a Model.\n3- Settings >>\n4- Treat Model.',
            '\n5- Classificatioin >>\n\n0- Exit.\n\nChoose Your Option: ',end='')
            Choice=int(input())
            if (Choice==1): Menu.New_Model()
            elif (Choice==2): Menu.Load_Model()
            elif (Choice==3): Menu.Setting_Menu()
            elif (Choice==4):
                if (Menu.Active_Model): Menu.Treat_Menu()
                else: 
                    print('\n\nThere is no Active Model...\n\nPress Enter to Continue...',end='')
                    input()
            elif (Choice==5) : Menu.Calssification_Menu(Menu.DropOut, Menu.DropOut_Rate)
            elif (Choice==0): loop=False
            else : print('ERROR - Wrong Number...\n\nPlease Choose a Correct Number...\n\n')    
#############################################################
    def Setting_Menu():
        loop=True
        while (loop):
            print('\n\t\t\t<<<<< Settings Menu >>>>>')
            print('\n\n1-Noises>>\n2-Change DropOut To',not Menu.DropOut,'\n\n0-Back')
            ch=int(input('\n\nPlease Choose Your Option : '))
            if (ch==1): NM.Noise_Menu()
            elif (ch==2): 
                Menu.DropOut = not(Menu.DropOut)
                if (Menu.DropOut):
                    print('\nDropOut Now Change to', Menu.DropOut,'\n\nPlease Enter DropOut Rate (Current Rate is',Menu.DropOut_Rate,'): ',end='')
                    Menu.DropOut_Rate=float(input())
            else : loop=False
#############################################################
    def New_Model():#By Choosing a Function Start a New Model To Learn
        os.chdir(Menu.root)
        os.chdir('models')
        os.chdir(Menu.NaFuncFoldModels)
        np.random.seed(0)
        ModelName=input('\n\nPlease Enter New Model''s Name : ')
        os.mkdir(ModelName)
        os.chdir(ModelName)
        NeuralNet.Model_Directory=os.getcwd()
        NeuralNet.Model_Name=ModelName
        print('Choose Function that You Want Make Data Sets For:')
        print('\n\n\n 1- F(x) = x^2\n 2- F(x) = Sinx \n 3- F(x) = Tanx\n 4- F(x) = Sqrt(x)\n 5- F(x) = |Cos(x)|\n 6- F(x) = [x]'
              '\n 7- Discrete -- F(x) = Sqrt(x) & x^2-2\n 8- F(x) = (x^2+1)/(x^2-4)\n 9- F(x) = |Ln(x)|\n 10- F(x) = x^3'
              '\n 11- F(x) = log2(x) '
              ' \n 0- Exit\n\nChoose a Number:',end='')
        Fnumber=int(input())
        if Fnumber==1: 
            Filename='x2TrainSet.csv'
            NeuralNet.Function_N=1
            ytitle='x^2'
        elif Fnumber==2:
            Filename='SinTrainSet.csv'
            NeuralNet.Function_N=2
            ytitle='Sin(x)'
        elif Fnumber==3: 
            Filename='TanTrainSet.csv'
            NeuralNet.Function_N=3
            ytitle='Tan(x)'
        elif Fnumber==4: 
            Filename='sqrtTrainSet.csv'
            NeuralNet.Function_N=4
            ytitle='SQRT(x)'
        elif Fnumber==5: 
            Filename='AbsCosTrainSet.csv'
            NeuralNet.Function_N=5
            ytitle='|Cos(x)|' 
        elif Fnumber==6:
            Filename='FloorTrainSet.csv'
            NeuralNet.Function_N=6
            ytitle='Floor(x)'
        elif Fnumber==7:
            Filename='DiscreteTrainSet.csv'
            NeuralNet.Function_N=7
            ytitle='Discrete   Sqrt(x) & x^2-1'
        elif Fnumber==8:
            Filename='RationalTrainSet.csv'
            NeuralNet.Function_N=8
            ytitle='(x^2+1)/(x^2-4)' 
        elif Fnumber==9:
            Filename='AbsLNTrainSet.csv'
            NeuralNet.Function_N=9
            ytitle='|Ln(x)|' 
        elif Fnumber==10:
            Filename='x3TrainSet.csv'
            NeuralNet.Function_N=10
            ytitle='x^3' 
        elif Fnumber==11:
            Filename='LogTrainSet.csv'
            NeuralNet.Function_N=11
            ytitle='Log(x)' 
        else : 
            os.chdir(Menu.root)
            os.chdir('models')
            os.chdir(Menu.NaFuncFoldModels)
            shutil.rmtree(ModelName) 
            return(0)
        NeuralNet.Model_TrainSetFilename=Filename
        print('\n\nPlease Enter the Number of Train Set :',end='')
        TrainSetN=int(input())
        NeuralNet.N_TrainigSet=TrainSetN
        DSM.Data_Maker(Filename, Fnumber, TrainSetN)
        TestFileName=DSM.NewName_TestSet(Filename)#Choose differnt name for test data ser file based on training dataset filename
        NeuralNet.Noise_Function_N, NeuralNet.Noise_Percentage = DSM.Add_Noise(Filename)
        NxTrain, x_train, y_train, NxTest, x_test, y_test = NeuralNet.Load_Network_Data(Filename, TestFileName)#بارگذاری مجموعه داده‌های آموزشی
        Title = 'Trainig with '+str(NxTrain)+' Data set '#+str(NoisePerc)+'% Malicious Noise'
        print('\nPlease Enter the Iteration of Neural Network Learnings: ',end='')
        Iteration=int(input())
        NeuralNet.N_Iteration=Iteration
        NeuralNet.Run_Neural_Network(Iteration, x_train, y_train, x_test, y_test, NxTrain, NxTest, Title, ytitle, True)
        Menu.Active_Model=True
#############################################################
    def Load_Model():
        os.chdir(Menu.root)
        os.chdir('models')
        os.chdir(Menu.NaFuncFoldModels)
        Menu.Models=os.listdir()
        if len(Menu.Models)==0:
            print('There is No Saved Model...\nPress Enter To Continue... ',end='')
            input()
            Menu.Active_Model=False
            return()
        else : 
            Model=NeuralNet.Load_Network()
        if Model==0 : 
            print('Error in Loading Model...\nPress Enter To Continue...',end='')
            input()
            Menu.Active_Model=False
        else:
            ans=input('\n\nDo You want Use New Test Data Set(y/n)? ')
            if ans=='y' or ans=='Y' : NewData=True
            else : NewData=False
            x_test, y_test = NeuralNet.LoadModel_TestDataMaker(NewData)
            NeuralNet.Use_Model(Model, x_test, y_test)
            scores=0
            print('\n\n')
            for i in range(1000):
                print('Running', i+1,'/1000 Test...',end='\r')
                x_test, y_test = NeuralNet.LoadModel_TestDataMaker(True)
                scores = scores + Model.evaluate(x_test, y_test, verbose=0)[0]
            print('average of 1000 run is :',scores/1000,'                   ')
            input()
            Menu.Active_Model=True
        return(Model, x_test, y_test)
#############################################################
    def NewNoisePercentage(NBeforeT,NInjec,NoiseP):
        NoiseN = DSM.Round_Percentage(NBeforeT, NoiseP)
        return((NoiseN/(NBeforeT+NInjec))*100)
#############################################################
    def Treat_Menu():
        os.chdir(NeuralNet.Model_Directory)
        loop=True
        while loop:
            print('\n\t\t\t<<<<< Treat Menu >>>>>')
            print('Here are Suggested Solution For Addressing Noise:\n\n1- Duplicate the Dataset in Test File.\n2- Inject Especific Amounts',
                  ' of True Data into Training Dataset File.\n3- Load Previously Saved Intact Model.\n\nPlease Choose Your Option: ',end='')
            Choice=int(input())
            if Choice==1:
                TestFileName=DSM.NewName_TestSet(NeuralNet.Model_TrainSetFilename)
                Duplicate_Filename=Treatment.Duplicate_Training_DataSet(NeuralNet.Model_TrainSetFilename)  
                NxTrain, x_train, y_train, NxTest, x_test, y_test = NeuralNet.Load_Network_Data(Duplicate_Filename, TestFileName)  
                ytitle=NeuralNet.ModelF
                if NeuralNet.Noise_Percentage!=0 : 
                    NeuralNet.Noise_Percentage=Menu.NewNoisePercentage(NeuralNet.N_TrainigSet,NeuralNet.N_TrainigSet,NeuralNet.Noise_Percentage)
                NeuralNet.N_TrainigSet=NeuralNet.N_TrainigSet*2
                Title = 'Trainig with '+str(NeuralNet.N_TrainigSet)+' Data set, Which is Duplicated '
                predict=NeuralNet.Run_Neural_Network(NeuralNet.N_Iteration, x_train, y_train, x_test, y_test, NxTrain, NxTest, Title, ytitle, False)                
            elif Choice==2: 
                print('\nThe Model Name is : ', NeuralNet.Model_Name,'\nThis Model Function is : F(x) =', NeuralNet.ModelF,'and It is Trained by', NeuralNet.N_TrainigSet,'Data in',NeuralNet.N_Iteration,'Iterations'
                      '\nThe Model has',NeuralNet.Noise_Percentage,'%',NeuralNet.NoiseFN)
                NInjD=int(input('\n\nHow Many True Data You Want to Inject into Trainig Dataset : '))
                Treatment.Inject_TrueData(NeuralNet.Model_TrainSetFilename,NInjD,NeuralNet.Function_N)
                if NeuralNet.Noise_Percentage!=0 : 
                    NeuralNet.Noise_Percentage=Menu.NewNoisePercentage(NeuralNet.N_TrainigSet,NeuralNet.N_TrainigSet+NInjD,NeuralNet.Noise_Percentage)
                NeuralNet.N_TrainigSet=NeuralNet.N_TrainigSet+NInjD
                ytitle=NeuralNet.ModelF
                Title = 'Training with '+str(NeuralNet.N_TrainigSet)+' Data set that '+str(NInjD)+' are Injected Data'
                TestFileName = DSM.NewName_TestSet(NeuralNet.Model_TrainSetFilename)
                NxTrain, x_train, y_train, NxTest, x_test, y_test = NeuralNet.Load_Network_Data(NeuralNet.Model_TrainSetFilename, TestFileName)
                predict=NeuralNet.Run_Neural_Network(NeuralNet.N_Iteration, x_train, y_train, x_test, y_test, NxTrain, NxTest, Title, ytitle, False)
            elif Choice==3:
                print('Not Writed!!')
            elif Choice==0: loop=False
            else : print('\n\nERROR- Wrong Number...\n\n')
        return(0)
#############################################################
    def Calssification_Menu(DropOut, DropOut_Rate):
        Function_Classifier.DropOut=DropOut
        Function_Classifier.DropOut_Rate=DropOut_Rate
        loop=True
        while loop:
            os.system('cls')
            print('\n\t\t\t<<<<< Classification Menu >>>>>')
            print('\n\n1-Pre Leaning Classifier>>\n2-Non Pre Learning Classifier>>\n3-Settings>>\n\n0-Back')
            Choice=int(input('\n\nPlease Choose Your Option : '))
            if Choice==1 : Menu.PrelearningClassifier_Menu()
            elif Choice==2 : Menu.NonPrelearningClassifier_Menu()
            elif Choice==3 : Menu.Setting_Menu()##Just Another Access For Comfort
            else: loop=False
#############################################################
    def PrelearningClassifier_Menu():
        Function_Classifier.DropOut=Menu.DropOut
        Function_Classifier.DropOut_Rate=Menu.DropOut_Rate
        loop=True
        while loop:
            os.system('cls')
            os.chdir(Menu.root)
            os.chdir('Models')
            os.chdir(Menu.NaPreLearModels)
            print('\n\t\t\t<<<<< Pre Learning Classifier Menu >>>>>')
            print('\n\n1-Makeing New Project Data\n2-Use Saved Classifier\n\n0-Back')
            Choice=int(input('\n\nPlease Choose Your Option : '))
            if Choice==1 : Function_Classifier.New_Project_PreLearn()
            elif Choice==2 : Function_Classifier.PreLearning_Classifier()
            else: loop=False
#############################################################
    def NonPrelearningClassifier_Menu():
        loop=True
        NPLCpath = Menu.root + '\Models\\' + Menu.NaNonPreLearModels
        while (loop) : 
            os.system('cls')
            os.chdir(NPLCpath) 
            print('\n\t\t\t<<<<< Non Pre Learning Classifier Menu >>>>>')          
            print('\n\n1- New Model\n2- Load Model\n3- Constant Data >>\n\n0-Back\n\n\n Please Choose Your Option: ',end='')
            ch =int(input())
            if ch==1 : 
                Function_Classifier.DropOut=Menu.DropOut
                Function_Classifier.DropOut_Rate=Menu.DropOut_Rate
                Function_Classifier.NonPrelearning_Classifier(NPLCpath)
            elif ch==2 : 
                os.chdir(Menu.root+'\\Models\\'+Menu.NaNonPreLearModels)
                Model, ModelName = Function_Classifier.Load_NPLC()
                os.chdir(Menu.root+'\\Models\\'+Menu.NaNonPreLearModels)
                Function_Classifier.Use_NPLC_Model(Model, ModelName)
            elif ch==3 : Menu.ConstantData_NPLC_Menu()
            elif ch==0 : loop=False
            else : print('\nError-Wrong Number...')
#############################################################
    def ConstantData_NPLC_Menu():
        loop=True
        ConsNPLCpath = Menu.root + '\Models\\' + Menu.Constant
        ConsModelPath = Menu.root + '\Models\\' + Menu.ConsModel
        while (loop) : 
            os.system('cls')
            os.chdir(ConsNPLCpath)
            print('\n\t\t\t<<<<< Non Pre Learning Classifier Constant Data Menu >>>>>')          
            print('\n\n1- Creat New Data\n2- Use Data \n3- Use Saved Model\n4- Settings>>\n\n0-Back\n\n\n Please Choose Your Option: ',end='')
            ch =int(input())
            if ch==1 : Function_Classifier.Making_CData_For_NPLC(ConsNPLCpath)
            elif ch==2 : 
                Function_Classifier.DropOut=Menu.DropOut
                Function_Classifier.DropOut_Rate=Menu.DropOut_Rate
                Function_Classifier.Models_by_CData(ConsNPLCpath)
            elif ch==3 : Function_Classifier.Use_ConsModel(ConsModelPath)
            elif ch==4 : Menu.Setting_Menu()
            elif ch==0 : loop=False
            else : print('\nError-Wrong Number...')
#############################################################
m=Menu
m.Main_Menu()
##############################################################