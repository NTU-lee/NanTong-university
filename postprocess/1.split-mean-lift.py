# -*- coding: utf-8 -*-
import numpy as np
import os
import random
import pandas as pd
import csv

import time

#

def fileList(filePath):
    newlist = [random.randint(0, 100) for i in range(100)]
    # print(len(newlist))
    fileLists = os.listdir(filePath)
    for flie in fileLists:
        flied = flie.split('.')
        if (flied[1] != '0'):
            newlist[int(flied[1]) - 1] =  flie
    return newlist


def sortP(file):
    data = np.genfromtxt('./all-data/'+file, delimiter=',', names=True, dtype='float16')
    # data1 =  data[np.argsort(data[:,0])]

    df = pd.DataFrame(data)
    # print(df)
    data = df.sort_index(axis=0, ascending=True, by=['Points1'])
    # data.to_csv('test.csv', index=0)
    # print(data)
    poinr0s = data['Points1'].unique()
    # poinr0s= np.argsort(poinr0s)
    # print(poinr0s)
    # print( ])
    for i in range(len(poinr0s)):
        a=os.path.exists('./new_data/'+str(i))
        # print(a==False)

        if a==False:
            os.makedirs('./new_data/' + str(i))


        # file.split('.')[0]+file.split('.')[1]
        data.loc[df['Points1'] == poinr0s[i]].to_csv('./new_data/'+str(i)+'/'+ file.split('.')[0]+file.split('.')[1]+ '.csv', index=0)

    #     d_data = data[data['Points1']].isin([poinr0])
    #     exec ("data%s.2f=d_data"%poinr0)
    # print(d_data)
    # data.to_csv('1.csv', index=0)
    # print((data))


if __name__ == "__main__":
    filePath = './all-data'
    fileList=fileList(filePath)
    print(fileList)
    for f in range(len(fileList)):
        # print(fileList[f])
        # data = np.genfromtxt(fileList[f], delimiter=',', names=True, dtype='float16')
        # data = pd.DataFrame(data)
        print(fileList[f])
        sortP(fileList[f])
    print '拆拆完成！'


def f(fileList):
    for i in range(len(fileList)):
        # with open(filePath + '/' + file, 'rb') as f:
        #     reader = csv.reader(f, delimiter=',')
        #     frame = pd.DataFrame(reader)
        #     print(reader)
        # print(filePath + '/' + fileList[i])
        data = np.genfromtxt(fileList[i], delimiter=',', names=True, )

        # col =np.array(col)
        x = data['Points0']
        y = data['Points2']
        m = (np.max(y) + np.min(y)) / 2
        alpha = np.pi - np.arctan2(y - m, x)

        df = pd.DataFrame(data)
        df = pd.concat([df, pd.DataFrame(alpha, columns=['alpha'])], axis=1)
        df=df.sort_index(axis=0, ascending=True, by=['alpha'])
        # print(df)
        # break
        df.to_csv(fileList[i], index=0)
        # frame = pd.DataFrame(data)
        # print(data['wallGradU0'])
def arv(filePath):
    newfileList = []
    fileList = os.listdir(filePath)
    # print(newfileList)
    for file in fileList:
        # print(file)
        filee = filePath + '/' + file
        newfileList.append(filee)
    f(newfileList)
    # print(df)
if __name__ == "__main__":
    filePath = './new_data'
    a = os.path.exists('./result')
    if a == False:
        os.makedirs('./result')
    for i in range(31):
        print('./result/' + str(i) )
        result = arv('./new_data/' + str(i))

    print('排序完成！')


def f(fileList, pram,totalp):

    for i in range(len(fileList)):
        # with open(filePath + '/' + file, 'rb') as f:
        #     reader = csv.reader(f, delimiter=',')
        #     frame = pd.DataFrame(reader)
        #     print(reader)
        # print(filePath + '/' + fileList[i])
        data = np.genfromtxt( fileList[i], delimiter=',', names=True, )
        # frame = pd.DataFrame(data)
        # print(data['wallGradU0'])
        # print(fileList[i])
        for ii in range(len(data[pram])):
            totalp[i][ii] = data[pram][ii]

    return totalp



def arv(filePath):
    # totalp = [[] for i in range(258)]
    # totalp = [[0 for col in range(258)] for row in range(101)]

    newfileList=[]
    fileList = os.listdir(filePath)
    # print(newfileList)
    for file in fileList:
               # print(file)
         filee=filePath+'/'+file
         newfileList.append(filee)

    dataone = np.genfromtxt(newfileList[0], delimiter=',', names=True, )
    ind=np.shape(dataone)[0]
    arv = np.zeros((4, ind))
    totalp = np.zeros((len(fileList), ind))
    totalp=f(newfileList,pram='p',totalp=totalp)
    df = pd.DataFrame(totalp)
    for a in range(len(df.columns)):
        # print("该列数据的均值位%.5f" % df[col].mean())  # 计算每列均值
        # print df.columns[a]
        arv[0][a] = df[a].mean()
        # arv[0][a] = get_rms(df[a])
    #     wallGradU:0
    total0 = f(newfileList, pram='wallShearStress0',totalp=totalp)
    df0 = pd.DataFrame(total0)
    for a in range(len(df0.columns)):
        # print("该列数据的均值位%.5f" % df[col].mean())  # 计算每列均值
        # print df.columns[a]
        arv[1][a] = df0[a].mean()
        # arv[1][a] = get_rms(df0[a])
        #     wallGradU:1
    total1 = f(newfileList, pram='wallShearStress1',totalp=totalp)
    df1 = pd.DataFrame(total1)
    for a in range(len(df1.columns)):
        # print("该列数据的均值位%.5f" % df[col].mean())  # 计算每列均值
        # print df.columns[a]
        arv[2][a] = df1[a].mean()
        # arv[2][a] = get_rms(df1[a])
      #     wallGradU:2
    total2 = f(newfileList, pram='wallShearStress2',totalp=totalp)
    df2= pd.DataFrame(total2)
    for a in range(len(df2.columns)):
        # print("该列数据的均值位%.5f" % df[col].mean())  # 计算每列均值
        # print df.columns[a]
        arv[3][a] = df2[a].mean()
        # arv[3][a] = get_rms(df2[a])
    return arv
    # print(df)


if __name__ == "__main__":
    filePath = './new_data'
    a = os.path.exists('./result')

    if a == False:
        os.makedirs('./result')
    for i in range(31):
        result = arv('./new_data/'+str(i))
        # print './new_data/'+str(i)

        result = pd.DataFrame(result).T
        print('./result/'+str(i)+'.csv')
        result.to_csv('./result/'+str(i)+'.csv',  index=0,header=['p','wallShearStress0','wallShearStress1','wallShearStress2'])


    print('平均完成！')
    
#    D = 0.0381
#U0 = 0.275
#nu = 10**-6
#Re = U0*D/nu
#
#def filelist(path):
#    fileList = []
#    for i in range(1, 101):
#        fileList.append(path + '/50' + str(i) + '.csv')
#    return fileList
#    # print(fileList)
#def doData(fileList):
#    list1=[]
#    list2 = []
#    for oneFile in fileList:
#        data = np.genfromtxt(oneFile, delimiter=',', names=True)
#        x = data['Points0']
#        y = data['Points2']
#        p = data['p']
#        vxs = data['wallShearStress0']
#        vzs = data['wallShearStress2']
#
#        m = (np.max(y)+np.min(y))/2
#        alpha =np.pi-np.arctan2(y-m, x)
#        drag = (np.sum(p*np.cos(alpha)+vxs)*np.pi)/(0.5*U0**2*1000)
#        list1.append(drag)
#        lift = (np.sum(p*np.sin(alpha)+vzs)*np.pi)/(0.5*U0**2*1000)
#        list2.append(lift)
#
#    return  np.array(list1), np.array(list2)
#
#if __name__ == "__main__":
#    filePath = './new_data'
#    alllist1=np.zeros([100,31])
#    alllist2 = np.zeros([100, 31])
#    for i in range(31):
#        fileList = filelist('./new_data/' + str(i))
#
#        list1,list2=doData(fileList)
#        alllist1[:,i]=list1
#        alllist2[:, i] = list2
#        print(i)
#        # print(list1)
#    df1 = pd.DataFrame(alllist1)
#    df2 = pd.DataFrame(alllist2)
#    df1.to_csv('drag.csv', index=1, )
#    df2.to_csv('ppp.dat', index=1, )
#    print('drag,lift完成！')
#
#with open('ppp.dat','rb') as f:
#    reader1 = csv.reader(f, delimiter =',')
#    c = list(reader1)
#    frame = pd.DataFrame(c) .T
#    frame.to_csv('out_data.dat', header=0,index=0)
#    print (frame)
#print("运行完成")