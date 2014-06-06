# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:40:20 2014

Edited on Fri June 6 13:45 2014 - updated the steps in the proceedure, as I was calculating the magnitudes incorrectly 
in the earlier version

@author: dicy

Compute the photometric parallax when colors and proper motions are known. The code is fairly simple, but I think it 
looks a little cluttered. The first portion opens the files, then a series of functions that are needed are defined, 
then the values from all the input files are passed thru the functions. Finally I make a plot. I'd like to move the plot
to another script then call the script at the end, but I'm not completely sure how I want to do that, so I'm open to
suggestions.

"""
import numpy as np
from pylab import figure, show

ifile1 = '/Users/dicy/Drive/GSUResearchandClasses/Research/KeplerWork/OutputFiles/2MASSSColorsSlowRotators.txt'
ifile2 = '/Users/dicy/Drive/GSUResearchandClasses/Research/KeplerWork/OutputFiles/2MASSSColorsFastRotators.txt'

ifile3 = '/Users/dicy/Drive/GSUResearchandClasses/Research/KeplerWork/OutputFiles/ProperMotionFastRotators.txt'
ifile4 = '/Users/dicy/Drive/GSUResearchandClasses/Research/KeplerWork/OutputFiles/ProperMotionSlowRotators.txt'

ifile5 = '/Users/dicy/Drive/GSUResearchandClasses/Research/KeplerWork/OutputFiles/DeclinationFastRotators.txt'
ifile6 = '/Users/dicy/Drive/GSUResearchandClasses/Research/KeplerWork/OutputFiles/DeclinationSlowRotators.txt'

ifile7 = '/Users/dicy/Drive/GSUResearchandClasses/Research/KeplerWork/OutputFiles/SDSSColorsSlowRotators.txt'
ifile8 = '/Users/dicy/Drive/GSUResearchandClasses/Research/KeplerWork/OutputFiles/SDSSColorsFastRotators.txt'

ifile9 = '/Users/dicy/Drive/GSUResearchandClasses/Research/KeplerWork/OutputFiles/mdwarfstandards.txt'

dslow = np.loadtxt(ifile1,delimiter=',')
dfast = np.loadtxt(ifile2,delimiter=',')

d2slow = np.loadtxt(ifile4,delimiter=',')
d2fast = np.loadtxt(ifile3,delimiter=',')

d3slow = np.loadtxt(ifile6,delimiter=',')
d3fast = np.loadtxt(ifile5,delimiter=',')

d4fast = np.loadtxt(ifile8,delimiter=',')
d4slow = np.loadtxt(ifile7,delimiter=',')

dstandards = np.loadtxt(ifile9, delimiter=',')

jslow = dslow[:,1]
hslow = dslow[:,2]
kslow = dslow[:,3]
jfast = dfast[:,1]
hfast = dfast[:,2]
kfast = dfast[:,3]

pmraFast = d2fast[:,1]
pmdecFast = d2fast[:,2]
pmraSlow = d2slow[:,1]
pmdecSlow = d2slow[:,2]

decSlow = d3slow[:,1]
decFast = d3fast[:,1]

gslow = list(d4slow[:,1])
rslow = list(d4slow[:,2])
#islow = d4slow[:,3]
gfast = list(d4fast[:,1])
rfast = list(d4fast[:,2])
#ifast = list(d4fast[:,3])

vStand = list(dstandards[:,0])
colStand = list(dstandards[:,1])
sortedColor = sorted(colStand)
plxStand = dstandards[:,2]

def cleanup(limit,inList,searchList):
    for item in inList[:]:
        if item <= limit:
            idx = inList.index(item)
            inList.pop(idx)
            searchList.pop(idx)
    return inList, searchList 
    
def cleanup2(limit,inList,searchList):
    for item in inList[:]:
        if item >= limit:
            idx = inList.index(item)
            inList.pop(idx)
            searchList.pop(idx)
    return inList, searchList     

#j,h,k are 2MASS magnitudes
#V is app mag
#M is abs mag 
#piPhot is photometric parallax
#mu is proper motion
#pmra and pmdec are prop motion in ra and dec, respectively
#g,r,i are SDSS mag

def transform(g,r):
    V = (g+r)/2.0
    return V
    #gotta look for something... hmmm
   
limit2 = 2
colStand,vStand = cleanup(limit2,colStand,vStand)  
limit = 20
vStand,colStand = cleanup2(limit,vStand,colStand)
print(max(vStand))  
    
def photometricparallax(M,V):
    piPhot = 10**((M-V-5)/5)
    return piPhot    
    
def color(V,k):
    col = V-k
    return col    
    
def dist(piP):
    dist = 1/piP
    return dist
    
#def distancemodulus(M,V):
 #   dmod = V-M
  #  return dmod    
    
def absmag(V,dist):
    Mabs = V - 5*np.log10(dist) + 5
    return Mabs
    
def mu(pmra,pmdec,declination):
    declination = (np.pi*declination)/180
    mu = pmdec*pmdec + pmra*pmra*np.cos(declination)*np.cos(declination)
    mu = np.sqrt(mu)
    return mu

def component(mu,dist):
	vcomp = 4.74*mu*dist
	return vcomp
    
def transverse(mu,dist):
    vtrans = 4.74*mu*dist
    return vtrans         
    
Vslow = [transform(a,b) for a,b in zip(gslow,rslow)]    
Vfast = [transform(d,e) for d,e in zip(gfast,rfast)]

colSlow = [color(aa,bb) for aa,bb in zip(Vslow,jslow)]
colFast = [color(aa,bb) for aa,bb in zip(Vfast,jfast)]

distStand = [dist(ddd) for ddd in plxStand]
Mstand = [absmag(eee,fff) for eee,fff in zip(vStand,distStand)]

coeff = np.polyfit(colStand,Mstand,1)

def transform2(color):
    M = np.polyval(coeff,color)
    return M
    #look in henry et al 2004  

Mslow = [transform2(m) for m in colSlow]
Mfast = [transform2(p) for p in colFast]

colSlow, Mslow = cleanup(limit2,colSlow,Mslow)
colFast, Msfast = cleanup(limit2,colFast,Mfast)

piSlow = [photometricparallax(s,t) for s,t in zip(Mslow,Vslow)]
piFast = [photometricparallax(x,y) for x,y in zip(Mfast,Vfast)]

#dmodSlow = [distancemodulus(mm,nn) for mm,nn in zip(Mslow,Vslow)]
#dmodFast = [distancemodulus(pp,qq) for pp,qq in zip(Mfast,Vfast)]

muSlow = [mu(xx,yy,zz) for xx,yy,zz in zip(pmraSlow,pmdecSlow,decSlow)]
muFast = [mu(aaa,bbb,ccc) for aaa,bbb,ccc in zip(pmraFast,pmdecFast,decFast)]

distSlow = [dist(ss) for ss in piSlow]
distFast = [dist(tt) for tt in piFast]

vraSlow = [component(iii,jjj) for iii,jjj in zip(pmraSlow,distSlow)]
vraFast = [component(kkk,mmmm) for kkk,mmmm in zip(pmraFast,distFast)]
vdecSlow = [component(nnn,ppp) for nnn,ppp in zip(pmdecSlow,distSlow)]
vdecFast = [component(ggg,hhh) for ggg,hhh in zip(pmdecFast,distFast)]

vtransSlow = [transverse(cc,dd) for cc,dd in zip(muSlow,distSlow)]
vtransFast = [transverse(ee,ff) for ee,ff in zip(muFast,distFast)]

fig = figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
fig.subplots_adjust(hspace=0.5)

ax1.plot(colStand,Mstand,'c.')
ax1.plot(sortedColor,np.polyval(coeff,sortedColor),'r-')
ax1.plot(colSlow,Mslow,'b.')
ax1.plot(colFast,Mfast,'m.')
ax1.set_ylim(0,20)
#ax1.set_xlim(-150,150)
ax1.set_ylim(ax1.get_ylim()[::-1])
ax1.set_xlabel(r'V-J')
ax1.set_ylabel(r'$M_V$')

ax2.plot(distSlow,vtransSlow,'b.')
ax2.plot(distFast,vtransFast,'m.')
ax2.set_xlabel(r'Distance (pc)')
ax2.set_ylabel(r'$v_T (km s^{-1})$')

ax3.plot(vraSlow,vdecSlow,'b.')
ax3.plot(vraFast,vdecFast,'m.')
ax3.set_xlabel(r'$v_\alpha (km s^{-1})$')
ax3.set_ylabel(r'$v_\delta$ $(km s^{-1})$')
show()
