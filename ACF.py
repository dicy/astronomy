# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:32:49 2013

@author: dicy
"""

def auto_correlate(x,y):
    """
    Takes a time series of magnitudes and range of lags to test and returns the auto correlation 
    function for the series. This algorithm is based on McQuillan, 2013, MNRAS. I wrote this code 
    to recreate the work presented in their paper for educational purposes. Our version of the ACF is:
    r_k = sum[(y_i^2*y_ik^2)/(y_i^2+y_ik^2)]
    ACF_k = 2*mean(r_k)
    where yi is the ith value of the time series, and yik is the value kth ahead in the series.
    """
    
    #get the ranges we'll be using, typically use N/2
    N = len(x)
    laglimit = N
    lagrange = range(0,laglimit)    
    
    y2 = [n*n for n in y]
    
    #calc the ACF
    ACF = []
    tauk = []
    for k in lagrange:
        print k
        stop = N - k
        rk = []
        for i in range(1,stop):
            index = i+k
            Si = y2[i]
            Sik = y2[index]
            sumSiSik = Si + Sik
            if sumSiSik > 0:
                yik = (y[i]*y[index])/sumSiSik
                rk.append(yik)
        if len(rk)==0:
            ACFhere=0
        else:
            ACFhere = 2*mean(rk)
        ACF.append(ACFhere)
        tauk.append(k)
 
    #get tauk into units of time rather than number of lags
    cadance = x[2]-x[1]
    tauk = [u*cadance for u in tauk]
    
    return tauk,ACF
            
from numpy import loadtxt
import csv
from itertools import izip   

starFiles = open('removed for secuirty.txt').readlines()
#the input file is just a text file filled with the paths to the files for which I wanted to compute the ACF
stripList = [i.strip() for i in starFiles]
stripList = stripList[:-1]
print stripList

starNames = []
for b in stripList[:-1]:
    starName = b.split('_')[1]
    print starName
    starName = starName.split('.')[0]
    print starName
    starNames.append(starName)

noOfStars = len(starNames)

for j in range(noOfStars):
    ifile = stripList[j]
    data = loadtxt(ifile,delimiter=',')
    starHere = starNames[j]
    print starHere
    outfile = "/Users/dicy/Drive/GSUResearchandClasses/Research/KeplerWork/OutputFiles/ACF_%s.csv"%starHere
    
    jd = data[:,0]
    flux = data[:,3]

    tauk,ACF = auto_correlate(jd,flux)
    
    with open(outfile, 'wb') as f:
        writer = csv.writer(f)
        #col headers would be periods strength
        writer.writerows(izip(tauk,ACF))
        close(outfile)


