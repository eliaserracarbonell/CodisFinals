#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:22:55 2024

@author: eserra
"""

# FLOOD FILL (NO PAR) VS GET ACCEPTANCE (PAR I NO PAR) I TRACK (PAR I NOPAR)

import numpy as np
import matplotlib.pyplot as plt
import at
import at.plot
import time
import h5py


ring = at.load_mat('ring.mat')


def floodfill (ring, nvoltes, delta, step):

    xvals = np.arange(-10, 10.1, step)
    yvals = np.arange(5, -5.1, -step) 
    punts = [(y,x) for y in yvals for x in xvals] # d'esquerra a dreta, de dalt a baix
    nx = len(xvals)
    npart = len(punts) # =nx*ny
    coord = np.reshape(punts, (npart,2))
    coord = np.transpose(coord)
    closed = at.find_orbit(ring) # no poso find_orbit6 perquè així si vull ring.disable_6d no ho he de canviar
    #closed = at.find_orbit6(ring, dp=delta) # Aquest delta és el offset d'energia de la closed orbit. Des/commentar aquesta línia al contrari que 31 i 36
    particules = np.zeros((6,npart))
    particules[0] = coord[1]*1e-3 + 1e-5
    particules[2] = coord[0]*1e-3 + 1e-5
    particules[4] = delta # Això és la variació d'energia de les partícules sobre la closed (on ja està sumat el offset incial!!)
    #particules += closed[0].reshape(6,1) # al fer això també estic sumant les coord x,y... de la closed (en algun cas la x és no despreciable i el gràfic es mou una mica cap a un costat!)
    particules[5] = closed[0][5]
    
    cua = [] # guardarà les partícules a explorar
    cua.append(0) # hi afegim la primera partícula (extrem superior esquerra)
    fets = [] # mantenim compte de les partícules que ja hem tracked
    noperd = np.empty((2,0)) # guardarà les partícules no perdudes que floodfill ha explorat
    st = time.time()
    while len(cua)!=0:
        i = cua.pop(0) # Traiem de la cua el primer element i a continuació l'analitzem:
        if (i >= 0) and (i < npart) and (i not in fets):
            fets.append(i)
            if (ring.track(particules[:,i], nturns=nvoltes, losses=True))[2]['loss_map']['islost'][0] == True: # Si està perduda
                cua.append(i+1) # explorem la partícula de la dreta
                cua.append(i-1) # partícula de l'esquerra
                cua.append(i+nx) # partícula de sota
                cua.append(i-nx) # partícula de dalt
            else: # Si no està perduda i hi hem "arribat", és part de la "frontera"
                noperd = np.hstack((noperd, ([[particules[0][i]], [particules[2][i]]])))
    temps = round((time.time()-st)/60, 1)
    
    exp = np.empty((2,0)) # Partícules explorades (les perdudes + les frontera)
    noexp = np.empty((2,0)) # Partícules no explorades (s'entén que la majoria són no perdudes)
    for i in range(0, npart):
        if i in fets:
            exp = np.hstack((exp, ([[particules[0][i]], [particules[2][i]]])))
        else:    
            noexp = np.hstack((noexp, ([[particules[0][i]], [particules[2][i]]])))
    
    plt.figure(figsize=(16,8))
    plt.scatter(exp[0]*1e3, exp[1]*1e3, color='lightgrey')
    plt.scatter(noexp[0]*1e3, noexp[1]*1e3, color='white')
    plt.scatter(noperd[0]*1e3, noperd[1]*1e3, color='firebrick')
    plt.title('Dynamic Aperture with Flood-Fill.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%, step='+str(step)+'mm, time='+str(temps)+'min')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.savefig('da_ff_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'.jpeg')
    # Bo per a latex: fig.savefig('nom.eps', format='eps', dpi=1200) (però amb el previsualitzador de l'ordinador no es veu massa bé)
    
    # Ara per a trobar la frontera de DA ens quedem amb els punts noperd més propers a l'origen
    radis = np.sqrt(noperd[0]**2 + (2*noperd[1])**2) # multiplico y per 2 per a fer-ho un cercle no aixafat
    angles = np.arctan2(2*noperd[1], noperd[0])
    numangles = 360
    rangtheta = np.linspace(-np.pi, np.pi, numangles, endpoint=False)
    frontera = np.empty((2,0))
    for theta in rangtheta:
        marge = 2*np.pi / numangles  # Divideixo el cercle en sectors
        sector = (angles >= theta - marge) & (angles < theta + marge)
        if np.any(sector): 
            idx_candidats = np.where(sector)[0] # els punts noperd dins el sector
            idx_minim = idx_candidats[np.argmin(radis[idx_candidats])] # escullo el més proper a l'origen
            frontera = np.hstack((frontera, ([[noperd[0][idx_minim]], [noperd[1][idx_minim]]])))
    
    #Aquí hauria d'afegir lo de moving average per a descartar els punts que s'escapen
    #dist = np.sqrt(frontera[0]**2 + frontera[1]**2)
    #np.convolve(dist, np.ones(N)/N, mode='valid')
    #...
    
    plt.figure(figsize=(16,8))
    plt.scatter(exp[0]*1e3, exp[1]*1e3, color='lightgrey')
    plt.scatter(noexp[0]*1e3, noexp[1]*1e3, color='white')
    plt.scatter(frontera[0]*1e3, frontera[1]*1e3, color='firebrick')
    plt.title('Dynamic Aperture Border, Flood-Fill.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%, step='+str(step)+'mm, time='+str(temps)+'min')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.savefig('da_ffboundary_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'.jpeg')

    name = 'da_ff_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'.hdf5'
    f = h5py.File(name,'w')
    mp = f.create_group('floodfill')
    mp.create_dataset('turns',data=np.array([nvoltes]))
    mp.create_dataset('dp',data=np.array([delta]))
    mp.create_dataset('step',data=np.array([step]))
    mp.create_dataset('execution_time',data=np.array([temps]))
    mp.create_dataset('explored',data=exp)
    mp.create_dataset('explored_notlost',data=noperd)
    mp.create_dataset('notexplored',data=noexp)
    mp.create_dataset('boundary',data=frontera)
    f.close()          
            
    #return (exp, noexp, noperd, frontera)


def getacc (ring, nvoltes, delta, step, mp):
    nx = int(1+(10*2)/step)
    ny = int(1+(5*2)/step)
    st = time.time()
    frontera,notlost,totes = at.get_acceptance(ring, ['x','y'], [nx,ny], [0.01,0.005], nturns=nvoltes, dp=delta, bounds=((-1, 1), (-1, 1)), grid_mode=at.GridMode.CARTESIAN, use_mp=mp, verbose=False)
    temps = round((time.time()-st)/60, 1)
    plt.figure(figsize=(16,8))
    plt.scatter(totes[0]*1e3, totes[1]*1e3, color='lightgrey')
    plt.scatter(notlost[0]*1e3, notlost[1]*1e3, color='black')
    plt.plot(frontera[0]*1e3, frontera[1]*1e3, color='firebrick', linewidth=3)
    plt.title('Dynamic Aperture with at.get_acceptance.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%, step='+str(step)+'mm, use_mp='+str(mp)+', time='+str(temps)+'min')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.savefig('da_ga_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'_'+str(mp)[0]+'.jpeg')


def track (ring, nvoltes, delta, step, mp):
    xvals = np.arange(-10, 10.1, step) 
    yvals = np.arange(-5, 5.1, step) 
    punts = [(x,y) for x in xvals for y in yvals]
    npart = len(punts)
    coord = np.reshape(punts, (npart,2))
    coord = np.transpose(coord)
    closed = at.find_orbit(ring)
    particules = np.zeros((6,npart))
    particules[0] = coord[0]*1e-3 + 1e-5
    particules[2] = coord[1]*1e-3 + 1e-5
    particules[4] = delta
    particules[5] = closed[0][5]
    lost = np.empty((2,0))
    notlost = np.empty((2,0))
    st = time.time()
    temp = ring.track(particules, nturns=nvoltes, losses=True, use_mp=mp)
    for i in range (0,npart):
        if temp[2]['loss_map']['islost'][i] == True:
            lost = np.hstack((lost, ([[particules[0][i]], [particules[2][i]]])))
        else:
            notlost = np.hstack((notlost, ([[particules[0][i]], [particules[2][i]]])))
    temps = round((time.time()-st)/60, 1)
    plt.figure(figsize=(16,8))
    plt.scatter(lost[0]*1e3, lost[1]*1e3, color='lightgrey')
    plt.scatter(notlost[0]*1e3, notlost[1]*1e3, color='black')
    plt.title('Dynamic Aperture with at.track.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%, step='+str(step)+'mm, use_mp='+str(mp)+', time='+str(temps)+'min')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.savefig('da_tr_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'_'+str(mp)[0]+'.jpeg')
    

def floodfillpx (ring, nvoltes, delta, step): 
    xvals = np.arange(-10, 10.1, step)
    pxvals = np.arange(1, -1.01, -step/5)
    punts = [(px,x) for px in pxvals for x in xvals] # d'esquerra a dreta, de dalt a baix
    nx = len(xvals)
    npart = len(punts)
    coord = np.reshape(punts, (npart,2))
    coord = np.transpose(coord)
    closed = at.find_orbit(ring)
    particules = np.zeros((6,npart))
    particules[0] = coord[1]*1e-3 + 1e-5 # x
    particules[1] = coord[0]*1e-3 # px
    particules[2] = 1e-5 
    particules[4] = delta
    particules[5] = closed[0][5]
    cua = [] 
    cua.append(0)
    fets = []
    noperd = np.empty((2,0))
    st = time.time()
    while len(cua)!=0:
        i = cua.pop(0)
        if (i >= 0) and (i < npart) and (i not in fets):
            fets.append(i)
            if (ring.track(particules[:,i], nturns=nvoltes, losses=True))[2]['loss_map']['islost'][0] == True:
                cua.append(i+1)
                cua.append(i-1)
                cua.append(i+nx)
                cua.append(i-nx)
            else:
                noperd = np.hstack((noperd, ([[particules[0][i]], [particules[1][i]]])))
    temps = round((time.time()-st)/60, 1)   
    exp = np.empty((2,0))
    noexp = np.empty((2,0))
    for i in range(0, npart):
        if i in fets:
            exp = np.hstack((exp, ([[particules[0][i]], [particules[1][i]]])))
        else:    
            noexp = np.hstack((noexp, ([[particules[0][i]], [particules[1][i]]])))
    plt.figure(figsize=(16,8))
    plt.scatter(exp[0]*1e3, exp[1]*1e3, color='lightgrey')
    plt.scatter(noexp[0]*1e3, noexp[1]*1e3, color='white')
    plt.scatter(noperd[0]*1e3, noperd[1]*1e3, color='firebrick')
    plt.title('Dynamic Aperture with Flood-Fill.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%, step='+str(step)+'mm, time='+str(temps)+'min')
    plt.xlabel('x (mm)')
    plt.ylabel('px (mrad)')
    plt.savefig('da_ff_px_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'.jpeg')
    radis = np.sqrt(noperd[0]**2 + (10*noperd[1])**2) # multiplico els px per 10 per a donar-los-hi més pes
    angles = np.arctan2(10*noperd[1], noperd[0]) # si no, tenim una el·lipse molt aixafada, i els punts de la frontera no són equidistants
    numangles = 360
    rangtheta = np.linspace(-np.pi, np.pi, numangles, endpoint=False)
    frontera = np.empty((2,0))
    for theta in rangtheta:
        marge = 2*np.pi / numangles  # Divideixo el cercle en sectors
        sector = (angles >= theta - marge) & (angles < theta + marge)
        if np.any(sector): 
            idx_candidats = np.where(sector)[0] # els punts noperd dins el sector
            idx_minim = idx_candidats[np.argmin(radis[idx_candidats])] # escullo el més proper a l'origen
            frontera = np.hstack((frontera, ([[noperd[0][idx_minim]], [noperd[1][idx_minim]]])))
    plt.figure(figsize=(16,8))
    plt.scatter(exp[0]*1e3, exp[1]*1e3, color='lightgrey')
    plt.scatter(noexp[0]*1e3, noexp[1]*1e3, color='white')
    plt.scatter(frontera[0]*1e3, frontera[1]*1e3, color='firebrick')
    plt.title('Dynamic Aperture Border, Flood-Fill.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%, step='+str(step)+'mm, time='+str(temps)+'min')
    plt.xlabel('x (mm)')
    plt.ylabel('px (mrad)') 
    plt.savefig('da_ffboundary_px_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'.jpeg')
    name = 'da_ff_px_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'.hdf5'
    f = h5py.File(name,'w')
    mp = f.create_group('floodfill')
    mp.create_dataset('turns',data=np.array([nvoltes]))
    mp.create_dataset('dp',data=np.array([delta]))
    mp.create_dataset('step',data=np.array([step]))
    mp.create_dataset('execution_time',data=np.array([temps]))
    mp.create_dataset('explored',data=exp)
    mp.create_dataset('explored_notlost',data=noperd)
    mp.create_dataset('notexplored',data=noexp)
    mp.create_dataset('boundary',data=frontera)
    f.close()
   
    
def floodfillpy (ring, nvoltes, delta, step): 
    yvals = np.arange(-6, 6.1, step/1.5) # OBS: l'step que entro i surt al títol no es correspon exactament amb el de les partícules
    pyvals = np.arange(1.6, -1.61, -step/2.5) 
    punts = [(py,y) for py in pyvals for y in yvals]
    ny = len(yvals)
    npart = len(punts)
    coord = np.reshape(punts, (npart,2))
    coord = np.transpose(coord)
    closed = at.find_orbit(ring)
    particules = np.zeros((6,npart))
    particules[2] = coord[1]*1e-3 +1e-5 # y
    particules[3] = coord[0]*1e-3 # py
    particules[0] = 1e-5
    particules[4] = delta
    particules[5] = closed[0][5]
    cua = [] 
    cua.append(0)
    fets = []
    noperd = np.empty((2,0))
    st = time.time()
    while len(cua)!=0:
        i = cua.pop(0)
        if (i >= 0) and (i < npart) and (i not in fets):
            fets.append(i)
            if (ring.track(particules[:,i], nturns=nvoltes, losses=True))[2]['loss_map']['islost'][0] == True:
                cua.append(i+1)
                cua.append(i-1)
                cua.append(i+ny)
                cua.append(i-ny)
            else:
                noperd = np.hstack((noperd, ([[particules[2][i]], [particules[3][i]]])))
    temps = round((time.time()-st)/60, 1)   
    exp = np.empty((2,0))
    noexp = np.empty((2,0))
    for i in range(0, npart):
        if i in fets:
            exp = np.hstack((exp, ([[particules[2][i]], [particules[3][i]]])))
        else:    
            noexp = np.hstack((noexp, ([[particules[2][i]], [particules[3][i]]])))
    plt.figure(figsize=(16,8))
    plt.scatter(exp[0]*1e3, exp[1]*1e3, color='lightgrey')
    plt.scatter(noexp[0]*1e3, noexp[1]*1e3, color='white')
    plt.scatter(noperd[0]*1e3, noperd[1]*1e3, color='firebrick')
    plt.title('Dynamic Aperture with Flood-Fill.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%, step='+str(step)+'mm, time='+str(temps)+'min')
    plt.xlabel('y (mm)')
    plt.ylabel('py (mrad)')
    plt.savefig('da_ff_py_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'.jpeg')
    radis = np.sqrt(noperd[0]**2 + (3*noperd[1])**2)
    angles = np.arctan2(3*noperd[1], noperd[0])
    numangles = 300
    rangtheta = np.linspace(-np.pi, np.pi, numangles, endpoint=False)
    frontera = np.empty((2,0))
    for theta in rangtheta:
        marge = 2*np.pi / numangles
        condicio_angular = (angles >= theta - marge) & (angles < theta + marge)
        if np.any(condicio_angular): 
            indexs_candidats = np.where(condicio_angular)[0]
            index_minim = indexs_candidats[np.argmin(radis[indexs_candidats])]
            frontera = np.hstack((frontera, ([[noperd[0][index_minim]], [noperd[1][index_minim]]])))
    plt.figure(figsize=(16,8))
    plt.scatter(exp[0]*1e3, exp[1]*1e3, color='lightgrey')
    plt.scatter(noexp[0]*1e3, noexp[1]*1e3, color='white')
    plt.scatter(frontera[0]*1e3, frontera[1]*1e3, color='firebrick')
    plt.title('Dynamic Aperture Border, Flood-Fill.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%, step='+str(step)+'mm, time='+str(temps)+'min')
    plt.xlabel('y (mm)')
    plt.ylabel('py (mrad)')
    plt.savefig('da_ffboundary_py_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'.jpeg')
    
    
def getaccpy (ring, nvoltes, delta, step, mp):
    ny = int(1+(6*2)/(step/1.5))
    npy = int(1+(1.6*2)/(step/2.5))
    st = time.time()
    frontera,notlost,totes = at.get_acceptance(ring, ['y','yp'], [ny,npy], [0.006,0.0016], nturns=nvoltes, dp=delta, bounds=((-1, 1), (-1, 1)), grid_mode=at.GridMode.CARTESIAN, use_mp=mp, verbose=False)
    temps = round((time.time()-st)/60, 1)
    plt.figure(figsize=(16,8))
    plt.scatter(totes[0]*1e3, totes[1]*1e3, color='lightgrey')
    plt.scatter(notlost[0]*1e3, notlost[1]*1e3, color='black')
    plt.plot(frontera[0]*1e3, frontera[1]*1e3, color='firebrick', linewidth=3)
    plt.title('Dynamic Aperture with at.get_acceptance.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%, step='+str(step)+'mm, use_mp='+str(mp)+', time='+str(temps)+'min')
    plt.xlabel('y (mm)')
    plt.ylabel('py (mrad)')
    plt.savefig('da_ga_py_'+str(nvoltes)+'_'+str(round(delta*1e2,2))+'_'+str(mp)[0]+'.jpeg')


# EXECUCIÓ
 
def comparativa (ring, nvoltes, delta, step): # tot i que potser fa petar l'ordinador
    floodfill(ring, nvoltes, delta, step)
    getacc(ring, nvoltes, delta, step, True)
    getacc(ring, nvoltes, delta, step, False)
    track(ring, nvoltes, delta, step, True)
    track(ring, nvoltes, delta, step, False)
        
floodfill(ring, 1000, 0, 0.1)  # 17 minuts.
getacc(ring, 1000, 0, 0.1, True) # 31 minuts
getacc(ring, 1000, 0, 0.1, False) # 74 minuts
track(ring, 1000, 0, 0.1, True)  # 26 minuts

floodfill(ring, 7000, 0, 0.1) # 1.5 hores. ho fa bé però em fa sortir uns missatges: "IOStream.flush timed out"
getacc(ring, 7000, 0, 0.1, True) # 3.5 hores
getacc(ring, 7000, 0, 0.1, False) # 8 hores

floodfillpy(ring, 1000, 0, 0.1) # 22 minuts
getaccpy(ring, 1000, 0, 0.1, True) # 18 minuts
getaccpy(ring, 1000, 0, 0.1, False) # 54 minuts

floodfillpy(ring, 7000, 0, 0.1) # 1.55 hores
getaccpy(ring, 7000, 0, 0.1, True) # 1.85 hores
getaccpy(ring, 7000, 0, 0.1, False) # 5.45 hores
