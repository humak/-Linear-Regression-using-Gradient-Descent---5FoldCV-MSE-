import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange
from random import seed
import sys

def readRegData():
    f = open("regression_data.txt","r")
    c1 = []
    c2 = []
    f1 = f.readline()   # skip the first line
    for line in f:
        f2 = [int(a) for  a in line.strip().split()]
        if f2!= []:
            x, y = f2
        else:
            continue
        c1 += [x]
        c2 += [y]
    return c1, c2

def readCT():
    f = open("classification_train.txt","r")
    feat1 = []
    feat2 = []
    l = []
    l0f1 = []
    l0f2 = []
    l1f1 = []
    l1f2 = []
    f1 = f.readline()   # skip the first line
    for line in f:
        f2 = [float(a) for  a in line.strip().split()]
        if f2!= []:
            x, y, z = f2
        else:
            continue
        feat1 += [x]
        feat2 += [y]
        l  += [z]
        if z == 0:
            l0f1 += [x]
            l0f2 += [y]
        if z == 1:
            l1f1 += [x]
            l1f2 += [y]
    return feat1, feat2, l, l0f1, l0f2, l1f1, l1f2



def gradientDescent(X, Y):
    w = 0
    b = 0
    rate = 0.00001  
    for i in range(100000): 
        guess = w*X + b  
        D_w = (-2/float(X.size)) * sum(X * (Y - guess))  # Derivative 
        D_b = (-2/float(X.size)) * sum(Y - guess)  
        w = w - rate * D_w  
        b = b - rate * D_b 
        error =( Y - guess ) * ( Y - guess )

    guess = w*X + b
    
    MSE = sum(error) / X.size
    #print (w, b, MSE)
    return w, b, MSE
    
def fiveFoldCV(X, Y):
    #print(X.size)
    foldSize = int(X.size / 5)
    #print(foldSize)
  
    trainSize = 4 * foldSize
    testSize = 1 * foldSize

    fold1 = X[0:foldSize]
    fold2 = X[foldSize:(foldSize*2)]
    fold3 = X[(foldSize*2):(foldSize*3)]
    fold4 = X[(foldSize*3):(foldSize*4)]
    fold5 = X[(foldSize*4):(foldSize*5)]

    fold1Y = Y[0:foldSize]
    fold2Y = Y[foldSize:(foldSize*2)]
    fold3Y = Y[(foldSize*2):(foldSize*3)]
    fold4Y = Y[(foldSize*3):(foldSize*4)]
    fold5Y = Y[(foldSize*4):(foldSize*5)]

    #print("test1")
    w1 = [None] * 6
    b1 = [None] * 6
    error1 = [None] * 6
    w1[2], b1[2], error1[2] = gradientDescent(fold2, fold2Y)
    w1[3], b1[3], error1[3]  = gradientDescent(fold3, fold3Y)
    w1[4], b1[4], error1[4]  = gradientDescent(fold4, fold4Y)
    w1[5], b1[5], error1[5]  = gradientDescent(fold5, fold5Y)
    MSE1 = ( error1[2] + error1[3] + error1[4] + error1[5] ) / 4
    #print(MSE1)
    #print("test2")
    w2 = [None] * 6
    b2 = [None] * 6
    error2 = [None] * 6
    w2[1], b2[1], error2[1] = gradientDescent(fold1, fold1Y)
    w2[3], b2[3], error2[3] = gradientDescent(fold3, fold3Y)
    w2[4], b2[4], error2[4] = gradientDescent(fold4, fold4Y)
    w2[5], b2[5], error2[5] = gradientDescent(fold5, fold5Y)
    MSE2 = ( error2[1] + error2[3] + error2[4] + error2[5] ) / 4
    #print(MSE2)
    ##print("test3")
    w3 = [None] * 6
    b3 = [None] * 6
    error3 = [None] * 6
    w3[1], b3[1], error3[1]  = gradientDescent(fold1, fold1Y)
    w3[2], b3[2], error3[2]  = gradientDescent(fold2, fold2Y)
    w3[4], b3[4], error3[4]  = gradientDescent(fold4, fold4Y)
    w3[5], b3[5], error3[5]  = gradientDescent(fold5, fold5Y) 
    MSE3 = ( error3[1] + error3[2] + error3[4] + error3[5] ) / 4
    #print(MSE3)
    ##print("test4")
    w4 = [None] * 6
    b4 = [None] * 6
    error4 = [None] * 6
    w4[1], b4[1], error4[1]  = gradientDescent(fold1, fold1Y)
    w4[2], b4[2], error4[2]  = gradientDescent(fold2, fold2Y)
    w4[3], b4[3], error4[3]  = gradientDescent(fold3, fold3Y)
    w4[5], b4[5], error4[5]  = gradientDescent(fold5, fold5Y) 
    MSE4 = ( error4[1] + error4[2] + error4[3] + error4[5] ) / 4
    ####print(MSE4)
    ###print("test5")
    w5 = [None] * 6
    b5 = [None] * 6
    error5 = [None] * 6
    w5[1], b5[1], error5[1]  = gradientDescent(fold1, fold1Y)
    w5[2], b5[2], error5[2]  = gradientDescent(fold2, fold2Y)
    w5[3], b5[3], error5[3]  = gradientDescent(fold3, fold3Y)
    w5[4], b5[4], error5[4]  = gradientDescent(fold4, fold4Y) 
    MSE5 = ( error5[1] + error5[2] + error5[3] + error5[4] ) / 4
    #####print(MSE5)
    print("PART1 b) 5-fold implemented.")

    overallMSE = (MSE1 + MSE2 + MSE3 + MSE4 + MSE5)/ 5
    print("PART1 c) Overall MSE: ",overallMSE)


def takeCov(x,y,meanx,meany,size):
    sumCov = 0
    for i in range (size):  
        sumCov = sumCov + (x[i]-meanx) * (y[i]-meany)
    return sumCov/size


def main():
    ''' PART 1 '''
    X, Y = np.array(readRegData())  # Head Size(x), Brain Weight(y)
    #print(x)
    #print(y)
    #plt.scatter(X, Y)
    #plt.show()
    for i in range (0, X.size):
        X[i] = X[i] / 1000
    for i in range (0, Y.size):
        Y[i] = Y[i] / 1000
    w, b, error = gradientDescent(X, Y)
    print("PART1 a) w: ",w," b: ",b)
    fiveFoldCV(X, Y)

    ''' PART 2 '''
    print("PART2 a)")
    f1, f2, l, l0f1, l0f2, l1f1, l1f2 = np.array(readCT())  # Head Size(x), Brain Weight(y)
    plt.scatter(f1, f2)
    plt.show() #a)
    pc1 = len(l0f1) / ( len(l0f1) + len(l1f1) )
    pc2 = len(l1f1) / ( len(l0f1) + len(l1f1) )
    print("P(C1) = ", pc1)
    print("P(C2) = ", pc2)
    print("PART2 b)")
    meanl0f1 = sum(l0f1) / len(l0f1)
    meanl0f2 = sum(l0f2) / len(l0f2)
    meanVector1 = np.array([meanl0f1, meanl0f2])
    print("Mean Vector 𝜇1 = ",meanVector1)

    cov11 = takeCov(l0f1,l0f1,meanl0f1,meanl0f1,len(l0f1))
    cov12 = takeCov(l0f1,l0f2,meanl0f1,meanl0f2,len(l0f1))
    cov21 = takeCov(l0f2,l0f1,meanl0f2,meanl0f1,len(l0f1))
    cov22 = takeCov(l0f2,l0f2,meanl0f2,meanl0f2,len(l0f1))

    covMat1 = ([cov11,cov12],[cov21,cov22])
    print("Covariance Matrix Σ1 = ",covMat1)

    meanl1f1 = sum(l1f1) / len(l1f1)
    meanl1f2 = sum(l1f2) / len(l1f2)
    meanVector2 = np.array([meanl1f1, meanl1f2])
    print("Mean Vector 𝜇2 = ",meanVector2)

    cov11_1 = takeCov(l1f1,l1f1,meanl1f1,meanl1f1,len(l1f1))
    cov12_1 = takeCov(l1f1,l1f2,meanl1f1,meanl1f2,len(l1f1))
    cov21_1 = takeCov(l1f2,l1f1,meanl1f2,meanl1f1,len(l1f1))
    cov22_1 = takeCov(l1f2,l1f2,meanl1f2,meanl1f2,len(l1f1))

    covMat2 = ([cov11_1,cov12_1],[cov21_1,cov22_1])
    print("Covariance Matrix Σ2 = ",covMat2)


    
if __name__ == "__main__":
    main()