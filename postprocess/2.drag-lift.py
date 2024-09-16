# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import csv

D = 0.0889
U0 = 0.2
nu = 10**-6
L = 0.14
Re = U0*D/nu

def filelist(path):
    fileList = []
    for i in range(1, 101):
        fileList.append(path + '/50' + str(i) + '.csv')
    return fileList
    # print(fileList)
def doData(fileList):
    list1=[]
    list2 = []
    for oneFile in fileList:
        data = np.genfromtxt(oneFile, delimiter=',', names=True)
        x = data['Points0']
        y = data['Points2']
        p = data['p']
        vxs = data['wallShearStress0']
        vzs = data['wallShearStress2']

        m = (np.max(y)+np.min(y))/2
        alpha =np.pi-np.arctan2(y-m, x)
        drag = (np.sum(p*np.cos(alpha)+vxs)*np.pi)/(0.5*U0**2*1000*D)*L
        list1.append(drag)
        lift = (np.sum(p*np.sin(alpha)+vzs)*np.pi)/(0.5*U0**2*1000*D)*L
        list2.append(-lift)

    return  np.array(list1), np.array(list2)

if __name__ == "__main__":
    filePath = './new_data'
    alllist1=np.zeros([100,31])
    alllist2 = np.zeros([100, 31])
    for i in range(31):
        fileList = filelist('./new_data/' + str(i))

        list1,list2=doData(fileList)
        alllist1[:,i]=list1
        alllist2[:, i] = list2
        print(i)
        # print(list1)
    df1 = pd.DataFrame(alllist1)
    df2 = pd.DataFrame(alllist2)
    df1.to_csv('drag.csv', index=1, )
    df2.to_csv('ppp.dat', index=1, )
    print('处理完成！')
    
with open('ppp.dat','rb') as f:
    reader1 = csv.reader(f, delimiter =',')
    c = list(reader1)
    frame = pd.DataFrame(c) .T
    frame.to_csv('out_data.dat', header=0,index=0)
    print (frame)

    # cut = './new_data/501.csv'
    # data = np.genfromtxt(cut, delimiter=',', names=True)
    # x = data['Points0']
    # y = data['Points2']
    # p = data['p']
    # vxs = data['wallShearStress0']
    # vzs = data['wallShearStress2']
    #
    # m = (np.max(y)+np.min(y))/2
    # alpha =np.pi-np.arctan2(y-m, x)
    # I1 = (np.sum(p*np.cos(alpha)+vxs)*np.pi)
    # I2 = (np.sum(p*np.sin(alpha)+vzs)*np.pi)
    # print(I1,I2)
