#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# eserra 2025feb11

import numpy as np
import at
import at.plot
import time
import h5py


#ring = at.load_mat('ALBA_II_20250129_divideSH1SV2SV4SV7_OCT_B_withapertures.mat')


def magnet_mod (ring, nturns, var=20, first_letter='S'):
    """Calculates the horizontal aperture limits with flood fill with varying sextupoles.

    ARGUMENTS:
    ring
    nturn
    KEYWORD ARGUMENTS:
    var : default 201
    """
    # get the sextupoles indexes
    print(f'var={var}, first_letter={first_letter}')
    idx_h = at.get_refpts(ring, first_letter + 'H*')
    idx_v = at.get_refpts(ring, first_letter + 'V*')
    if first_letter == 'S':
        iorder = 2
    if first_letter == 'O':
        iorder = 3
    #idx_S = np.concatenate((idx_sh[0:16], idx_sv[0:28]))
    idx_S = np.concatenate((idx_h, idx_v))
    #idx_oh = at.get_refpts(ring, 'OH*')
    #idx_ov = at.get_refpts(ring, 'OV*')
    #idx_S = np.concatenate((idx_oh, idx_ov))
    #iorder = 3
    idx_S = np.sort(idx_S)

    result = np.empty((0,6))
    st = time.time()

    thestep = 0.01
    neg, pos = floodfill (ring, nturns, step = thestep)
    minabsnegpos = min(abs(neg),pos)
    print(f'Aperture at start {1e3*minabsnegpos:0.4} mm')

    # iterate in a quarter
    lenidxquarter = int(len(idx_S)/4)
    maxaper = 0
    idxmaxaper = 0
    for idx in range(lenidxquarter):
        quarter_ring = [0,1,2,3]
        tmpval = [0,0,0,0]
        # go on every quadrant
        for quadrant in quarter_ring:
            ele = idx_S[idx + lenidxquarter*quadrant]
            if ring[ele].PolynomB[iorder] > 0:
                deltastep = +1*var
            else: # ring[ele].PolynomB[iorder] < 0
                deltastep = -1*var
            tmpval[quadrant] = ring[ele].PolynomB[iorder]
            ring[ele].PolynomB[iorder] += deltastep
            # print(ele)
        neg, pos = floodfill (ring, nturns, step = thestep)
        minabsnegpos = min(abs(neg),pos)
        result = np.vstack((result, ([idx, round(tmpval[quadrant],1), deltastep, neg*1e3, pos*1e3, minabsnegpos*1e3])))

        for quadrant in quarter_ring:
            ele = idx_S[idx + lenidxquarter*quadrant]
            ring[ele].PolynomB[iorder] = tmpval[quadrant]
        print(f'Evaluating idx {idx}, total {lenidxquarter}, famname {ring[idx_S[idx]].FamName}, minaperture = {minabsnegpos*1e3:0.4} mm')
        if minabsnegpos > maxaper:
            maxaper = minabsnegpos
            idxmaxaper = idx

    print(f'Max aperture with variations {1e3*maxaper:0.3} at idx {idxmaxaper}, FamName {ring[idx_S[idxmaxaper]].FamName}')
    temps = round((time.time()-st)/60, 1)

        
    name = 'sextupoles_variation_'+str(var)+'.hdf5'
    f = h5py.File(name,'w')
    mp = f.create_group('results')
    mp.create_dataset('nturns', data=np.array([nturns]))
    mp.create_dataset('execution_time (min)', data=np.array([temps]))
    mp.create_dataset('table', data=result)
    f.close() 
    print(f'Output filename: {name}')
    return result


def floodfill (ring, nturns, delta=0, step=0.1):
    #xvals = np.arange(-10, 10.1, step)
    xvals = np.arange(-7.5, 7.6, step)
    yvals = np.arange(1.5, -1.51, -3*step) # pocs valors per y perquè només ens interessa l'eix x
    yvals = np.array([-step, +step])
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
