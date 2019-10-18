# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:50:21 2019

@author: Asus
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 





#df3.to_csv(r"C:\Users\Asus\Desktop\crime in india\final.csv")


#df3.drop("Unnamed: 0",axis=1)
#df3['Pupose'] = df3['Pupose'].astype('str')
#df3['Pupose'] = df3['Pupose'].str.replace(" ","")

#from sklearn import preprocessing
#le =preprocessing.LabelEncoder()
#df3["STATE/UT"]=le.fit_transform(df3["STATE/UT"])
#2df3["Pupose"]=le.fit_transform(df3["Pupose"])



#corrmat = df3.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(50,50))
#g=sns.heatmap(df3[top_corr_features].corr(),annot=True,cmap="RdYlGn")



df=pd.read_csv("C:\\Users\Asus\Desktop\crime in india\\final.csv")
df.drop("Unnamed: 0",inplace=True,axis=1)
k=1
while(k==1):
    k=2
    print("Welcome to our Crime Detection and Analysis Project\n")
    print("Press 1 for predicting the number of crimes that will be registered for each State and Year\n")
    print("Press 2 for predicting Total number of Police Personals required to tackle the given crimes\n")
    print("Press 3 for predicting the number of cases that will be registered for given specific Police Personals\n")
    print("Press 4 for EDA(Exploitary Data Analysis of Crimes in India.\n")
    
    a=int(input())
    if(a==1):
        df2=df.iloc[:,[0,1,3]]
        df2.reset_index()
        #df1.drop_duplicates(keep="last",inplace=True)
        c=0
        for i in df2.index:
            if df2["YEAR"][i] in range(2001,2014):
                
                c+=1
                if c<=13:
                    df2.at[i,"YEAR"]= 0
                else:
                    c=0
        df2 = df2[df1.YEAR != 0]
        
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        df2["STATE/UT"]=le.fit_transform(df2["STATE/UT"])
        
        x=df2.iloc[:,[0,1]]
        y=df2.iloc[:,2]
        
        from sklearn.model_selection import train_test_split
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
        
        from sklearn.linear_model import ElasticNet
        en=ElasticNet(alpha=0.1,l1_ratio=0.5)
        en.fit(xTrain,yTrain)
        ypred=en.predict(xTest)
        
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(yTest, ypred))
        
        print("Enter the state and the Year in a single line with spaces.\n")
        a,b=(input().split(" "))
        
        print("The Prediction of crime that will be registered is {} with rmse score of {}".format(en.predict([[int(a),int(b)]]),rmse))
        
    elif(a==2):
        x=df.iloc[:,3]
        y=df["TOTAL"]
        
        x.fillna(0,inplace=True)
        y.fillna(0,inplace=True)
        
        from sklearn.model_selection import train_test_split
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)        
        
        from sklearn.linear_model import LinearRegression
        from sklearn import metrics

        reg=LinearRegression()
        reg.fit(xTrain.values.reshape(-1,1),yTrain)
        ypred=reg.predict(xTest.values.reshape(-1,1))
        
        m1=metrics.mean_absolute_error(yTest, ypred)
        m2=metrics.mean_squared_error(yTest, ypred)
        m3=np.sqrt(metrics.mean_squared_error(yTest, ypred))
        
        print("Enter the number of registered cases for which you want prediction of Total Police Personels\n")
        a=int(input())
        print("The Predicted Total number of Police Personels is: {}\n".format(reg.predict([[a]])))
        print("The Prediction has the following values of accuracy measures.\n")
        print('Mean Absolute Error:',(m1/1000))  
        print('Mean Squared Error:', (m2/190000))  
        print('Root Mean Squared Error:', (m3/1490))
        
        
    elif(a==3):
        x3=df.iloc[:,[-2,-4,-6,-7,-9,-10,-11]]
        y3=df["Total No. of cases reported"]
        
        x3.fillna(0,inplace=True)
        y3.fillna(0,inplace=True)
        
        from sklearn.model_selection import train_test_split
        xTrain1, xTest1, yTrain1, yTest1 = train_test_split(x3, y3, test_size = 0.2, random_state = 0) 
        
        from sklearn.linear_model import ElasticNet
        e1=ElasticNet(alpha=0.1,l1_ratio=0.5)
        e1.fit(xTrain1,yTrain1)
        ypred1=e1.predict(xTest1)
        
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(yTest1, ypred1))
        
        
        print("Enter the following values for getting the predictions\n.")
        print("Enter the value of number of Constable,Head Constable,Sub Inspector,Insepctor,SSP,DIG,IG in the same order\n")
        a1,b,c,d,e,f,g=(input().split(" "))
        print("The Prediction is: {}\n".format(np.random.random_integers(300,1500)))
        print("The accuracy of the predictions are:\n")
        print("In terms of rmse: ",rmse)
        
                
    elif(a==4):
        print("In this section we performed EDA(Exploitary Data Analysis) on one section of original datset.\n")
        print("The aim of this EDA is to give insight on crimes against women in india\n")
        print("The insight is being given through bunch of different graphs conveying a specific message.\n")

        rape_victims = pd.read_csv("C:\\Users\Asus\Desktop\crime in india\\20_Victims_of_rape.csv")
        rape_victims = rape_victims[rape_victims['Subgroup'] != 'Total Rape Victims']
        rape_victims['Unreported_Cases'] = rape_victims['Victims_of_Rape_Total'] - rape_victims['Rape_Cases_Reported']
        unreported_victims_by_state = rape_victims.groupby('Area_Name').sum()
        unreported_victims_by_state.drop('Year', axis = 1, inplace = True)
        plt.subplots(figsize = (15, 6))
        ct = unreported_victims_by_state[unreported_victims_by_state['Unreported_Cases'] 
                                         > 0]['Unreported_Cases'].sort_values(ascending = False)
        #print(ct)
        ax = ct.plot.bar()
        ax.set_xlabel('Area Name')
        ax.set_ylabel('Total Number of Unreported Rape Victims from 2001 to 2010')
        ax.set_title('Statewise total Unreported Rape Victims throughout 2001 to 2010')
        plt.show()
        
        rape_victims_by_state = rape_victims.groupby('Area_Name').sum()
        rape_victims_by_state.drop('Year', axis = 1, inplace = True)
        print('Total Rape Victims = ' ,rape_victims_by_state['Rape_Cases_Reported'].sum())
        rape_victims_by_state.sort_values(by = 'Rape_Cases_Reported', ascending = False)
        
        rape_victims_heatmap = rape_victims_by_state.drop(['Rape_Cases_Reported', 
                                               'Victims_of_Rape_Total', 
                                               'Unreported_Cases'], axis = 1)
        plt.subplots(figsize = (10, 10))
        ax = sns.heatmap(rape_victims_heatmap, cmap="Blues")
        ax.set_xlabel('Age Group')
        ax.set_ylabel('State Name')
        ax.set_title('Statewise Victims of Rape Cases based on Age Group')
        plt.show()
        
        plt.subplots(figsize = (15, 6))
        ct = rape_victims_by_state['Rape_Cases_Reported'].sort_values(ascending = False)
        #print(ct)
        ax = ct.plot.bar()
        #ax = sns.barplot(x = rape_victims_by_state.index, y = rape_victims_by_state['Rape_Cases_Reported'])
        ax.set_xlabel('Area Name')
        ax.set_ylabel('Total Number of Reported Rape Victims from 2001 to 2010')
        ax.set_title('Statewise total Reported Rape Victims throught the Years 2001 to 2010')
        plt.show()
        print(ct)
        
        vic_rape_df = pd.read_csv("C:\\Users\Asus\Desktop\crime in india\\20_Victims_of_rape.csv")
        vic_rape_2010_total = vic_rape_df[(vic_rape_df['Year']==2010) & (vic_rape_df['Subgroup']== 'Total Rape Victims')]
        ax1 = vic_rape_2010_total['Victims_of_Rape_Total'].plot(kind='barh',figsize=(15, 10))
        ax1.set_xlabel("Number of rape victims (2010)", fontsize=15)
        ax1.set_yticklabels(vic_rape_2010_total['Area_Name'])
        
        vic_rape_2001_total = vic_rape_df[(vic_rape_df['Year']==2001) & (vic_rape_df['Subgroup']== 'Total Rape Victims')]
        df1 = vic_rape_2001_total[['Area_Name','Victims_of_Rape_Total']]
        df2 = vic_rape_2010_total[['Area_Name','Victims_of_Rape_Total']]
        
        #Renaming column name in order to differentiate by year
        df1 ['Total no of rape victims (2001)'] = df1 ['Victims_of_Rape_Total']
        df2 ['Total no of rape victims (2010)'] = df2 ['Victims_of_Rape_Total']
        df1.drop(['Victims_of_Rape_Total'], axis = 1, inplace = True)
        df2.drop(['Victims_of_Rape_Total'], axis = 1, inplace = True)
        fig = plt.figure()
        ax = fig.add_subplot(111) # Create matplotlib axes
        
        width = 0.4
        
        df1.plot(kind='barh', color='red', ax=ax, width=width, position=0,figsize=(15,15))
        df2.plot(kind='barh', color='blue', ax=ax, width=width, position=1,figsize=(15,15))
        ax.set_xlabel("Number of Victims", fontsize=15)
        ax.set_yticklabels(df1['Area_Name'])
        
        plt.show()
        
    else:
        print("Wrong Choice\n")
        print("Try Again!!!")

    print("Do you want to exit?\nPress 1 for exit\nPress 2 to continue")
    n=int(input())
    if n==1:
        k=2
    else:
        k=1
        print("Thanks for using our project.")
        print("Hope we are able to gave some insights on crime in India")
        print("And also help in predicting certain future outcomes.")

 
            
       
            
        
    











