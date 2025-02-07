#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import at
import at.plot
import time
import h5py


ring = at.load_mat('ALBA_II_20250129_divideSH1SV2SV4SV7_OCT_B_withapertures.mat')


def sextupoles (ring, nturns, var=20):
    idx_sh = at.get_refpts(ring, 'SH*')
    idx_sv = at.get_refpts(ring, 'SV*')
    idx_S = np.concatenate((idx_sh[0:16], idx_sv[0:28]))
    idx_S = np.sort(idx_S)
    
    result = np.empty((0,6))
    st = time.time()
    for ele in idx_S:
        if ring[ele].PolynomB[2] > 0:
            ring[ele].PolynomB[2] -= var
            neg, pos = floodfill (ring, nturns)
            ring[ele].PolynomB[2] += var
            result = np.vstack((result, ([ele, round(ring[ele].PolynomB[2],1), -var, neg*1e3, pos*1e3, min(abs(neg),pos)*1e3])))
        else: # ring[ele].PolynomB[2] < 0
            ring[ele].PolynomB[2] += var
            neg, pos = floodfill (ring, nturns)
            ring[ele].PolynomB[2] -= var
            result = np.vstack((result, ([ele, round(ring[ele].PolynomB[2],1), +var, neg*1e3, pos*1e3, min(abs(neg),pos)*1e3])))
    temps = round((time.time()-st)/60, 1)
        
    name = 'sextupoles_variation_'+str(var)+'.hdf5'
    f = h5py.File(name,'w')
    mp = f.create_group('results')
    mp.create_dataset('nturns', data=np.array([nturns]))
    mp.create_dataset('execution_time (min)', data=np.array([temps]))
    mp.create_dataset('table', data=result)
        
    return result


def floodfill (ring, nturns, delta=0, step=0.1):
    xvals = np.arange(-10, 10.1, step)
    yvals = np.arange(0.5, -0.51, -step) # pocs valors per y perquè només ens interessa l'eix x
    punts = [(y,x) for y in yvals for x in xvals]
    nx = len(xvals)
    npart = len(punts)
    coord = np.reshape(punts, (npart,2))
    coord = np.transpose(coord)
    closed = at.find_orbit(ring)
    particules = np.zeros((6,npart))
    particules[0] = coord[1]*1e-3 + 1e-5
    particules[2] = coord[0]*1e-3 + 1e-5
    particules[4] = delta
    particules[5] = closed[0][5]
    
    notlost = np.empty((2,0)) # Partícules explorades no perdudes
    cua = [] # guardarà les partícules a explorar
    cua.append(0) # hi afegim la primera partícula
    fets = [] # guarda les partícules que ja hem tracked
    while len(cua)!=0:
        i = cua.pop(0)
        if (0 <= i < npart) and (i not in fets):
            fets.append(i)
            temp = (ring.track(particules[:,i], nturns=nturns, refpts=200, losses=True))[2]['loss_map']
            if temp['islost'][0] == True:
                cua.append(i+1)
                cua.append(i-1)
                cua.append(i+nx)
                cua.append(i-nx)
            else:
                notlost = np.hstack((notlost, ([[particules[0][i]], [particules[2][i]]])))
    
    # DA in x axis:
    radis = np.sqrt((notlost[0]-1e-5)**2 + (2*(notlost[1]-1e-5))**2)
    thetas = np.arctan2(2*(notlost[1]-1e-5), notlost[0]-1e-5)
    marge = np.pi / 180
    negsector = (thetas >= np.pi - marge) | (thetas < -np.pi + marge) # sector al voltant de -pi = pi
    possector = (thetas >= 0 - marge) & (thetas < 0 + marge) # sector al voltant de 0
    if np.any(negsector): 
        idx_candidats = np.where(negsector)[0]
        idx_minim = idx_candidats[np.argmin(radis[idx_candidats])]
        negative = notlost[0][idx_minim]
    if np.any(possector): 
        idx_candidats = np.where(possector)[0]
        idx_minim = idx_candidats[np.argmin(radis[idx_candidats])]
        positive = notlost[0][idx_minim]
    
    return negative, positive
