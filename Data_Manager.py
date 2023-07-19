# Load liabraries
import math 
import csv
import random
import numpy as np
import os
#############################################################
# Define a Class for makeing Data Set
class DataSets_Maker():
    def __init__(self):
        shuffle=False#Defualt Value of shuffle
#############################################################    
    def Round_Percentage(Num, Percentage):#Return Rounded Percentage of Given Nummber
        Perc=Num*Percentage/100
        if Num>=0:
            return(int(int(Perc*2+1)/2))
        else:
            return(int(int(Perc*2-1)/2))
#############################################################
    def NewName_TestSet(filename):#change Training set file into Test set file
        newname=filename.replace('Train','Test')
        return (newname)
#############################################################
    def Random_List(first,end,N):#Makeing a Random and unpretitive list between first and end number
        List=list()
        i=0
        while i<N:
            List.append(random.randint(first,end-1))
            List.sort()
            check=True
            j=0
            while (j<(len(List)-1)) & (check):
                if List[j]==List[j+1]:
                    List.remove(List[j])
                    N+=1
                    check=False
                else: j+=1
            i+=1
        return List
#############################################################
    def Noise_List(TraSetN, NoiP, shuffle):#by Given number of data set and percentage,make a list in which item should replaced by Noisy data
        Nlist=list()                       #in two way, Shuffled noisy data with main data or in ordered level
        if NoiP==0:
            for i in range(TraSetN):#without noise
                Nlist.append(1)
            return(Nlist)
        else:
            NoisyDataN=DataSets_Maker.Round_Percentage(TraSetN, NoiP)#determine number of noisy data that should made
            CorrectDataN=TraSetN-NoisyDataN
            if shuffle==False:#ordered append noisy data
                for i in range(CorrectDataN):
                   Nlist.append(1)#for real data
                for i in range(NoisyDataN):
                    Nlist.append(0)#for noisy data
                return(Nlist)
            else:
                random.seed(0)#shuffled noisy data
                rand_Shuff=DataSets_Maker.Random_List(0,TraSetN,NoisyDataN)
                j=0
                for i in range(TraSetN):
                    if (i==rand_Shuff[j]) :
                        Nlist.append(0)#for noisy data
                        if (j+1)<NoisyDataN: j+=1
                    else:
                        Nlist.append(1)#for real data
                return(Nlist)
#############################################################
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
            Pair.append(random.uniform(-5,9))
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
            P = np.random.choice([1,2], p=[0.55, 0.45])
            if P==1 : Pair.append(random.uniform(0.02,0.85))
            elif P==2 : Pair.append(random.uniform(0.85,5))
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
            Pair.append(random.uniform(0.1,10))
            Pair.append(np.log2(Pair[0]))
            return(Pair)
        return(0)
#############################################################
    def Add_Noise(FileName):#Ask For Adding Noise or Not
        print('\n\nDo You Want Add Noise To The Data(y/n)?',end='')
        Ans=input()
        Perc=0
        Noise_Name=0
        if Ans=='y' or Ans=='Y' : 
            File_Handeler=open(FileName)
            Reader=csv.reader(File_Handeler)
            Data=list(Reader)
            File_Handeler.close()
            y_list=list()
            for i in range(len(Data)):
                y_list.append(float(Data[i][1]))
            print('Here are Noises You Can Choose For Adding to Data Sets:\n\n1- Gaussian Noise\n2- Rayleigh Noise'
                   '\n3- Gamma Noise\n4- Exponential Noise\n5- Uniform Noise\n\nPlease Enter The Number of The Noise You Want '
                   'Add To Data Sets: ',end='')
            Noise_Name=int(input())
            print('Please Enter The Percentage of Noise You Want to Add(1-100) : ',end='')
            Perc=int(input())
            N=DataSets_Maker.Round_Percentage(len(Data),Perc)
            Noise_List=DataSets_Maker.Random_List(0,len(Data),N)
            for i in range(N):
                y_list[Noise_List[i]]=Noise_Management.Noise(y_list[Noise_List[i]], Noise_Name)
            for i in range(len(Data)):
                Data[i][1] = str(y_list[i])
            File_Handeler=open(FileName,'w',newline='')
            Writer=csv.writer(File_Handeler)
            Writer.writerows(Data)
            File_Handeler.close()
        return(Noise_Name, Perc)
#############################################################
    def Data_Maker(fname, FuncN, TrasN):#Makeing Data Sets
        File_Handeler=open(fname, "w", newline="")
        W=csv.writer(File_Handeler)
        for i in range(TrasN):
            TrainPair=list()
            TrainPair = DataSets_Maker.Make_Pair_Data(FuncN)
            W.writerow(TrainPair)
        File_Handeler.close()        
        testfile = DataSets_Maker.NewName_TestSet(fname)
        File_Handeler=open(testfile, "w", newline="")
        W=csv.writer(File_Handeler)
        TestN = DataSets_Maker.Round_Percentage(TrasN, 25)
        for i in range(TestN):
            TestPair=list()
            TestPair = DataSets_Maker.Make_Pair_Data(FuncN)
            W.writerow(TestPair)
        File_Handeler.close()
#############################################################
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
#############################################################
    def Make_Pair_Data_(FuncName):#Makeing a pair Data base on User Choises for correct or noisy data
        Pair=list()
        if FuncName==1:
            Pair.append(random.uniform(-3,3))
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
            Pair.append(random.uniform(-3, 3))
            Pair.append(Pair[0])
            return(Pair)
        elif FuncName==5:
            Pair.append(random.uniform(0,3))
            Pair.append(math.pow(Pair[0], 0.5))
            return(Pair)
        elif FuncName==6:
            Pair.append(random.uniform(-2,2))
            Pair.append(math.floor(Pair[0]))
            return(Pair)
        elif FuncName==7:
            Pair.append(random.uniform(-3,3))
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
                    x = random.uniform(-4,4)
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
        return(0)
#############################################################
####################################################### END OF CLASS #########################

############################################## BEGINIG OF CLASS ##############################
class Noise_Management():
    #This Class Mange Noises That Should Added to Data
    ##################################################
    ###### Noise Name Equvalent to Number: ###########
    ########## 1- Gaussian Noise #####################
    ########## 2- Rayleigh Noise #####################
    ########## 3- Gamma Noise ########################
    ########## 4- Exponential Noise ##################
    ########## 5- Uniform Noise ######################
    ##################################################
    ############## Set Noises Default Values##########
    G_Mu = 0       ####Guassian Noise Parameters µ,x in R and Seg^2>0
    G_S2 = 1
    ############
    Ray_S = 0.5     ####Rayleigh Noise Parameters Seg>0 and x in [0, inf)
    ############
    Gamma_a = 1    ####Gamma Noise Paramteres a>0 , b in N and x in [0, inf)
    Gamma_b = 2
    ############
    Exp_a = 1      ####Exponential Noise Parametes a>0 and x in [0, inf)      
    ############
    Unif_a = 0     ####Uniform Noise Parameters -inf<a<b<+inf on x in [a,b]
    Unif_b = 1
#############################################################
    def Add_Noise(Data, NoiseName, Percentage):
        N = DataSets_Maker.Round_Percentage(len(Data),Percentage)
        Noise_List = DataSets_Maker.Random_List(0,len(Data),N)
        for i in range(N):
            Data[Noise_List[i]][1] = Noise_Management.Noise(Data[Noise_List[i]][1], NoiseName)
        return(Data)
############################################################# 
    def Noise_Name(NoiseNumber):
        if NoiseNumber==1 : return('Gaussian')
        elif NoiseNumber==2 : return('Rayleigh')
        elif NoiseNumber==3 : return('Gamma')
        elif NoiseNumber==4 : return('Exponential')
        elif NoiseNumber==5 : return('Uniform')
        else : return('No')
#############################################################    
    def factorial(Number) : #Calculate Factorial of a Number
        f=1
        for i in range(1,Number+1) : 
            f=f*i
        return f
#############################################################
    def SetParameters(NoiseName) : # Set Parameters With Asking User
        if NoiseName==1 : #Gaussian Noise
            print('Plaese Enter µ : ',end='')
            Noise_Management.G_Mu=float(input())
            print('Plaese Enter Segma^2 :',end='') 
            Noise_Management.G_S2=float(input())
            while Noise_Management.G_S2<=0 : 
                print('Segma^2 Must be biger than 0...\n\n Please Enter a Valid Number : ',end='')
                Noise_Management.G_S2=float(input())
        elif NoiseName==2 : #Rayleigh Noise
            print('Please Enter Segma (Should Be >0): ',end='')
            Noise_Management.Ray_S=float(input())
            while Noise_Management.Ray_S<=0 : 
                print('Segma Must be >0...\n\n Please Enter a Valid Number : ',end='')
                Noise_Management.Ray_S=float(input())
        elif NoiseName==3 : #Gamma Noise
            print('Please Enter Alpha :',end='')
            Noise_Management.Gamma_a = int(input())
            print('Please Enter Beta (Must be in N): ',end='')
            Noise_Management.Gamma_b = int(input())
            while ((Noise_Management.Gamma_a<0) and (Noise_Management.Gamma_b<0)) :
                print('Please Enter Alpha :',end='')
                Noise_Management.Gamma_a = int(input())
                print('Please Enter Beta : ',end='')
                Noise_Management.Gamma_b = float(input())
        elif NoiseName==4 : #Exponential Noise
            print('Please Enter landa (>0) : ',end='')
            Noise_Management.Exp_a = float(input())
            while Noise_Management.Exp_a<=0 :
                print('Please Enter a Valid landa (>0) : ',end='')
                Noise_Management.Exp_a = float(input())
        elif NoiseName==5 : #Uniform Noise
            Noise_Management.Unif_a, Noise_Management.Unif_b = input('Please Enter a and b (a<b) :',end='').split()
            Noise_Management.Unif_a=float(Noise_Management.Unif_a)
            Noise_Management.Unif_b=float(Noise_Management.Unif_b)
            while Noise_Management.Unif_a==Noise_Management.Unif_b:
                Noise_Management.Unif_a, Noise_Management.Unif_b = input('Please Enter a and b (a<b) :',end='').split()
                Noise_Management.Unif_a=float(Noise_Management.Unif_a)
                Noise_Management.Unif_b=float(Noise_Management.Unif_b)
            if Noise_Management.Unif_a>Noise_Management.Unif_b : #change a with b
                Noise_Management.Unif_b = Noise_Management.Unif_b + Noise_Management.Unif_a
                Noise_Management.Unif_a = Noise_Management.Unif_b - Noise_Management.Unif_a
                Noise_Management.Unif_b = Noise_Management.Unif_b - Noise_Management.Unif_a
#############################################################
    def Noise(Input, NoiseName) : #Return Input added with Noise by Chosen Noise
        if NoiseName==1 : #Gaussian Noise
            return(Input+ (math.exp(-(math.pow(Input-Noise_Management.G_Mu,2)/(2*Noise_Management.G_S2*Noise_Management.G_S2)))/(math.sqrt(2*math.pi*Noise_Management.G_S2))))
        elif NoiseName==2 : #Rayleigh Noise
            if (Input<0) : 
                return(Input+0)
            else : 
                return(Input+ (Input/math.pow(Noise_Management.Ray_S,2))*math.exp(-math.pow(Input,2)/(2*math.pow(Noise_Management.Ray_S,2))))
        elif NoiseName==3 : #Gamma Noise
            if Input<0 : return(Input+0)
            else : 
                return(Input+ (math.pow(Noise_Management.Gamma_a,Noise_Management.Gamma_b)* math.pow(Input,Noise_Management.Gamma_b-1)*math.exp(-Noise_Management.Gamma_a*Input))/Noise_Management.factorial(Noise_Management.Gamma_b-1))
        elif NoiseName==4 : #Exponential Noise
            if Input<0 : return(Input+0)
            else:
                return(Input+ (Noise_Management.Exp_a*math.exp(-Noise_Management.Exp_a*Input)))
        elif NoiseName==5 : #Uniform Noise
            if (Noise_Management.Unif_a<Input<Noise_Management.Unif_b) : 
                return(Input+ (1/(Noise_Management.Unif_b-Noise_Management.Unif_a)))
            else : return(Input+0)
#############################################################
    def Select_Random_NoisyData(N_Data, Percentage) : #Make a Random list of Which Data should Injected Noise
        NoisyData_list=list()
        N_NoisyData=DataSets_Maker.Round_Percentage(N_Data,Percentage)
        while len(NoisyData_list)<N_NoisyData :
            R=random.randint(0,N_Data)
            if R not in NoisyData_list : NoisyData_list.append(R)
        return(NoisyData_list)
#############################################################
    def Noise_Menu():#Show and Could Change Noises Parameters
        loop=True
        while (loop):
            print('You Could Use 5 Different Noise to Add to the Data Sets.\n1- Gaussian Noise\n2- Raleigh Noise\n'
                  '3- Gamma Noise\n4- Exponential Noise\n5- Uniform Noise\n\nTo See and Set Noises Parameters Choose Noise Numbers(0 for Back) : ',end='')
            Choice=int(input())
            if Choice==1:
                print('Default Gaussian Noise Distribution Parameters are : \nMu= ',Noise_Management.G_Mu, 'Sigma^2= ',Noise_Management.G_S2,'\n'
                      '\nDo You Want Set New Parameters For Gaussian Noise(y/n)? ',end='')
                ans=input()
                if ans=='y' or ans=='Y' : Noise_Management.SetParameters(1)
            elif Choice==2:
                print('Default Rayleigh Noise Distribution Parameters are : \nSigma= ',Noise_Management.Ray_S,'\n'
                      '\nDo You Want Set New Parameters For Rayleigh Noise(y/n)? ',end='')
                ans=input()
                if ans=='y' or ans=='Y' : Noise_Management.SetParameters(2)
            elif Choice==3:
                print('Default Gamma Noise Distribution Parameters are : \nAlpha = ',Noise_Management.Gamma_a, 'Beta = ',Noise_Management.Gamma_b,'\n'
                      '\nDo You Want Set New Parameters For Gamma Noise(y/n)? ',end='')
                ans=input()
                if ans=='y' or ans=='Y' : Noise_Management.SetParameters(3)
            elif Choice==4:
                print('Default Exponential Noise Distribution Parameters are : \nLanda= ',Noise_Management.Exp_a,'\n'
                      '\nDo You Want Set New Parameters For Exponential Noise(y/n)? ',end='')
                ans=input()
                if ans=='y' or ans=='Y' : Noise_Management.SetParameters(4)
            elif Choice==5:
                print('Default Uniform Noise Distribution Parameters are : \na= ',Noise_Management.Unif_a, 'b= ',Noise_Management.Unif_b,'\n'
                      '\nDo You Want Set New Parameters For Uniform Noise(y/n)? ',end='')
                ans=input()
                if ans=='y' or ans=='Y' : Noise_Management.SetParameters(5)
            else: loop=False

####################################################### END OF CLASS #########################

############################################## BEGINIG OF CLASS ##############################
#Define a Class For Treatment of A Noisy Data Set 
class DataSet_Treatment():
    def Duplicate_Training_DataSet(fname):#Duplicate data of a sent file
        File_Reader=open(fname, 'r')
        Reader=csv.reader(File_Reader)
        Data=list(Reader)
        File_Reader.close()
        fname='Duplicate'+fname
        File_Duplicate=open(fname, "w", newline="")
        Writer=csv.writer(File_Duplicate)
        for i in  range(2):#Write Data into New File Twice
            Writer.writerows(Data)
        File_Duplicate.close()
        return(fname)
#############################################################    
    def Making_True_Data(Number, Fname):#Return a data list of true data 
        Data=list()
        if Fname==1:
            for i in range(Number):
                Pair=list()
                Pair.append(random.uniform(-10,10))
                Pair.append(math.pow(Pair[0], 2))
                Data.append(Pair)
            return(Data)
        elif Fname==2:
            for i in range(Number):
                Pair=list()
                Pair.append(random.uniform(0, 2*math.pi))
                Pair.append(math.sin(Pair[0]))
                Data.append(Pair)
            return(Data)
        elif Fname==3:
            for i in range(Number):
                Pair=list()
                Pair.append(random.uniform(-math.pi/2, math.pi/2))
                Pair.append(math.tan(Pair[0]))
                Data.append(Pair)
            return(Data)
        elif Fname==4:
            for i in range(Number):
                Pair=list()
                Pair.append(random.uniform(-10, 10))
                Pair.append(Pair[0])
                Data.append(Pair)
            return(Data)
        elif Fname==5:
            for i in range(Number):
                Pair=list()
                Pair.append(random.uniform(0,100))
                Pair.append(math.pow(Pair[0], 0.5))
                Data.append(Pair)
            return(Data)
#############################################################
    def Inject_TrueData(fname, NInjD, Fname):#Inject Correct Data to specific file data
        Filename=open(fname, "a+", newline="")
        Injector=csv.writer(Filename)
        Injection_List = list()
        for i in range(NInjD):
            NewPair=DataSets_Maker.Make_Pair_Data(Fname)
            Injector.writerow(NewPair)
        Filename.close()
#############################################################    
    def Add_Gauss_Noise(Data, Mu, Mean):# Add Huassian Noise With Specific Mu & Mean to Sent Data
        noise=np.random.normal(Mean,Mu,len(Data))            
        noise=noise.reshape((len(Data),1))
        NoisyِData = Data + noise
        return (NoisyِData)
#############################################################


#############################################################
####################################################### END OF CLASS #########################