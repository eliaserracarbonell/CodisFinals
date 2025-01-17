#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:13:06 2025

@author: eserra
"""

# FAST TOUSCHEK TRACKING

import numpy as np
import matplotlib.pyplot as plt
import at
import at.plot
import time
import h5py
import bisect


ring = at.load_mat('ALBA_II_20240326_A_5ba_3ss.mat')


# FUNCIÓ PRINCIPAL

def FastTouschekTracking (ring, nvoltes):
    st = time.time()
    momacc = np.empty((3,0)) # momentum acceptance
    dext = 0.04
    poly = dopolyhedron(ring, nvoltes, dext) # A s0, un sol cop.
    closedorbit = at.find_orbit6(ring, refpts=at.All)[1]
    opt = at.linopt6(ring) # si dopoly retorna coord no norm, aquestes dues linies no caldrien per a binarysearch
    binvB = np.array([[1, 0], [opt[0]['alpha'][0], opt[0]['beta'][0]]]) # acabar de mirar si és la matriu que toca!
    ndeltastep = -0.001
    pdeltastep = 0.001
    
    for i in range(1,len(ring)): # per a cada element de l'anell (1457)
        if ring[i].Length != 0: # (cada element rellevant) (1264)
            shortring = ring[i:]
            closed = closedorbit[i]
            na = 0; nb = 2**6 # low and high bounds
            pa = 0; pb = 2**6
            while nb-na > 1:
                na, nb = binarysearch(na, nb, ndeltastep, poly, shortring, closed, binvB)
                pa, pb = binarysearch(pa, pb, pdeltastep, poly, shortring, closed, binvB)
            dpN = na*ndeltastep # conservative estimates
            dpP = pa*pdeltastep
            momacc = np.hstack((momacc, ([[i], [dpN], [dpP]]))) 
    temps = round((time.time()-st)/3600, 1)   

    plt.figure(figsize=(16,8))
    plt.plot(momacc[0], momacc[1], linestyle='-', color='blue')
    plt.plot(momacc[0], momacc[2], linestyle='-', color='blue')
    plt.title('Momentum Acceptance, FTT.  nturns='+str(nvoltes)+', time='+str(temps)+'h')
    plt.xlabel('element index'); plt.ylabel('dp/p')
    plt.ylim(-dext-0.005,dext+0.005)
    plt.savefig('momentumacceptance_'+str(nvoltes)+'_.jpeg')

    name = 'momentumacceptance_'+str(nvoltes)+'.hdf5'
    f = h5py.File(name,'w')
    mp = f.create_group('momacc')
    mp.create_dataset('nturns', data=np.array([nvoltes]))
    mp.create_dataset('deltastep', data=np.array([pdeltastep]))
    mp.create_dataset('deltamax', data=np.array([dext]))
    mp.create_dataset('execution_time (h)', data=np.array([temps]))
    mp.create_dataset('dpP',data=momacc[2]) # Positive
    mp.create_dataset('dpN',data=momacc[1]) # Negative
    mp.create_dataset('index',data=momacc[0])
    f.close()

    return momacc


# FUNCIONS AUXILIARS

def dopolyhedron (ring, nvoltes, dext): 
    step = 0.1 
    xvals = np.arange(-10, 10.1, step)
    pxvals = np.arange(1, -1.01, -step/10)
    punts = [(px,x) for px in pxvals for x in xvals] # d'esquerra a dreta, de dalt a baix
    nx = len(xvals)
    npart = len(punts)
    coord = np.reshape(punts, (npart,2))
    coord = np.transpose(coord)
    closed = at.find_orbit(ring)
    particules = np.zeros((6,npart))
    particules[0] = coord[1]*1e-3 + 1e-5
    particules[1] = coord[0]*1e-3
    particules[2] = 1e-5
    particules[5] = closed[0][5]
    opt = at.linopt6(ring)
    binvB = np.array([[1, 0], [opt[0]['alpha'][0], opt[0]['beta'][0]]])
    
    ndeltes = int(1 + 2*dext/0.005) # /0.01 per deltes cada 1%, /0.005 per deltes cada 0.5%
    deltes = np.linspace(-dext, dext, ndeltes, endpoint=True)
    nangles = 180
    angles = np.linspace(-np.pi, np.pi, nangles, endpoint=False)
    prepoly = np.zeros((ndeltes,nangles,3)) 
    st = time.time()
    for i in range(0,ndeltes):
        prepoly[i] = floodfillpx(ring, nvoltes, particules, npart, nx, binvB, nangles, angles, deltes[i])
    
    nslices = int(1 + 2*dext/0.001) # resolució 10 / 5 vegades més fina que els deltes, respectivament
    slices = np.linspace(-dext, dext, nslices, endpoint=True)
    polyhedron = interpolate(prepoly, deltes, nslices, slices, nangles)
    temps = round((time.time()-st)/3600, 2)
    
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    for i in range(0, nslices):
        ax.scatter(polyhedron[i][:,0]*1e3,polyhedron[i][:,1]*1e3,polyhedron[i][:,2]*1e2, s=5, c='dodgerblue')
    plt.title('DA Polyhedron, with Flood-Fill and interpolation.  nturns='+str(nvoltes)+', time='+str(temps)+'h')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('Px (mm?)'); ax.set_zlabel('dp/p (%)')
    ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10]); ax.set_zlim([-dext*1e2, dext*1e2])
    ax.zaxis.labelpad=-1
    ax.view_init(elev=5, azim=-60, roll=0)
    plt.savefig('polyhedron_'+str(nvoltes)+'.jpeg')
    
    name = 'polyhedron_'+str(nvoltes)+'.hdf5'
    f = h5py.File(name,'w')
    mp = f.create_group('polyhedron')
    mp.create_dataset('nturns', data=np.array([nvoltes]))
    mp.create_dataset('ndeltas', data=np.array([ndeltes]))
    mp.create_dataset('nslices', data=np.array([nslices]))
    mp.create_dataset('nangles', data=np.array([nangles]))
    mp.create_dataset('execution_time (h)', data=np.array([temps]))
    mp.create_dataset('boundary', data=polyhedron)
    f.close()
    
    return polyhedron


def floodfillpx (ring, nvoltes, particules, npart, nx, binvB, nangles, angles, delta):
    particules[4] = delta

    noperd = np.empty((2,0))
    cua = [] 
    cua.append(0)
    fets = []
    st = time.time()
    while len(cua)!=0:
        i = cua.pop(0)
        if (0 <= i < npart) and (i not in fets):
            fets.append(i)
            if (ring.track(particules[:,i], nturns=nvoltes, losses=True))[2]['loss_map']['islost'][0] == True:
                cua.append(i+1)
                cua.append(i-1)
                cua.append(i+nx)
                cua.append(i-nx)
            else:
                noperd = np.hstack((noperd, ([[particules[0][i]], [particules[1][i]]])))
    temps = round((time.time()-st)/60, 1)   

    norm = binvB.dot(noperd) # Filtro partícules noperd amb radi mínim
    radis = np.sqrt((norm[0]-1e-5)**2 + norm[1]**2)
    thetas = np.arctan2(norm[1], norm[0]-1e-5)
    marge = np.pi / nangles
    prefrontera = np.empty((0,3))
    for theta in angles:
        if theta == angles[0]:
            sector = (thetas >= angles[-1] + marge) | (thetas < theta + marge)
        else:
            sector = (thetas >= theta - marge) & (thetas < theta + marge)
        if np.any(sector):
            idx_candidats = np.where(sector)[0]
            idx_minim = idx_candidats[np.argmin(radis[idx_candidats])]
            prefrontera = np.vstack((prefrontera, ([norm[0][idx_minim], norm[1][idx_minim], thetas[idx_minim]])))
    prefrontera = prefrontera[np.argsort(prefrontera[:,2])]
    
    plt.figure(figsize=(8,8))
    plt.scatter(prefrontera[:,0]*1e3, prefrontera[:,1]*1e3, color='firebrick')
    plt.title('Dynamic Aperture PREBorder, Flood-Fill.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%')
    plt.xlabel('x (mm)'); plt.ylabel('Px (mm ?)')
    plt.xlim(-10,10); plt.ylim(-10,10)
    
    frontera = np.empty((0,3)) # Col·loco partícules als angles que toquen. AIXÒ S'HA DE MILLORAR
    for theta in angles:
        idx = bisect.bisect_left(prefrontera[:,2], theta)
        if idx == 0 or idx == len(prefrontera):
            ang = [prefrontera[-1][2], prefrontera[0][2]+2*np.pi]
            lam = (abs(theta) - ang[0]) / (ang[1] - ang[0])
            idx = 0
        else:
            ang = [prefrontera[idx-1][2], prefrontera[idx][2]]
            lam = (theta - ang[0]) / (ang[1] - ang[0])
        x = (1-lam)*prefrontera[idx-1][0] + lam*prefrontera[idx][0]
        xp = (1-lam)*prefrontera[idx-1][1] + lam*prefrontera[idx][1]
        frontera = np.vstack((frontera, ([x, xp, delta]))) 

    plt.figure(figsize=(8,8))
    plt.scatter(frontera[:,0]*1e3, frontera[:,1]*1e3, color='firebrick')
    plt.title('Dynamic Aperture Border, Flood-Fill.  nturns='+str(nvoltes)+', dp='+str(round(delta*1e2,3))+'%, time='+str(temps)+'min')
    plt.xlabel('x (mm)'); plt.ylabel('Px (mm ?)')
    plt.xlim(-10,10); plt.ylim(-10,10)
    plt.savefig('xpx_boundary_'+str(nvoltes)+'_'+str(round(delta*1e2,3))+'.jpeg')
    
    return frontera


def interpolate (prepoly, deltes, nslices, slices, nangles):
    polyhedron = np.zeros((nslices,nangles,3))
    
    j=0
    for i in range(0,nslices):
        if slices[i] in deltes:
            polyhedron[i] = prepoly[j]
            j+=1
        else:
            idx = bisect.bisect_left(deltes, slices[i])
            lam = (slices[i] - deltes[idx-1]) / (deltes[idx] - deltes[idx-1])
            polyhedron[i][:,0] = (1-lam)*prepoly[idx-1][:,0] + lam*prepoly[idx][:,0]
            polyhedron[i][:,1] = (1-lam)*prepoly[idx-1][:,1] + lam*prepoly[idx][:,1]
            polyhedron[i][:,2] = slices[i]
    
    return polyhedron


def binarysearch (a, b, deltastep, poly, shortring, closed, binvB):                    
    m = (a+b)/2
    particula = np.array([closed[0]+1e-5, closed[1], closed[2]+1e-5, closed[3], closed[4]+m*deltastep, closed[5]])
    t = shortring.track(particula, nturns=1, losses=True)[0] # trasllat
    norm = binvB.dot(np.array([[t[0][0][0][0]], [t[1][0][0][0]]])) # normalitzem x, xp
    part = np.array([norm[0][0],norm[1][0],t[4][0][0][0]]) # només ens interessen x, xp i dp/p
    inside = particleinpoly(poly, part) 
    if inside: # == True
        return m, b
    else:
        return a, m
    
    
def particleinpoly (poly, part):
    nslices = len(poly)
    slices = poly[:,0,2]
    nangles = len(poly[0])
    angles = np.linspace(-np.pi, np.pi, nangles, endpoint=False)

    if part[2] < slices[0]-5e-4 or part[2] > slices[-1]+5e-4: 
        return False 
    
    i = bisect.bisect_left(slices, part[2]) # avaluo la partícula al polígon amb dp/p més proper (en v.abs)
    if (i == nslices) or (i != 0 and part[2]-slices[i-1] < slices[i]-part[2]):
        i = i-1
        
    phi = np.arctan2(part[1], part[0]-1e-5) 
    idx = bisect.bisect_left(angles, phi)
    if idx == nangles:
        ant = idx-1; post = 0 
    else:
        ant = idx-1; post = idx
    varx = poly[i][post][0] - poly[i][ant][0]
    varxp = poly[i][post][1] - poly[i][ant][1]
    if part[1]*varx - part[0]*varxp > poly[i][ant][1]*varx - poly[i][ant][0]*varxp:
        return True
    return False


# Per a calcular TOUSCHEK LIFETIME a partir del resultat de FTT:

def touscheklifetime (momacc):
    hcfactor = 3.9 
    pars = ring.radiation_parameters() 
    ex = pars.emittances[0]*1/(1 + pars.J[1]/pars.J[0])
    ey = ex
    ibunch = 0.75e-3
    sigmas = pars.sigma_l
    sigmasharmon = pars.sigma_l * hcfactor 
    sigmae = pars.sigma_e

    refpts = momacc[0].astype(int)
    momap = momacc[1:3].T # files dpN i dpP convertides en columnes
    tlwo, __ , __ = at.get_lifetime(ring,ey,ibunch,emitx=ex,sigs=sigmas,sigp=sigmae,momap=momap,refpts=refpts) # without harmon
    tlw, __ , __ = at.get_lifetime(ring,ey,ibunch,emitx=ex,sigs=sigmasharmon,sigp=sigmae,momap=momap,refpts=refpts) # with harmon
    
    return np.array([round(tlwo/3600,2),round(tlw/3600,2)]) # en hores