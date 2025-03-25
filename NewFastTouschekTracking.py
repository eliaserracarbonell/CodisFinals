"""FAST TOUSCHEK TRACKING algorithm amb "parallelized" floodfill"""

import numpy as np
import matplotlib.pyplot as plt
import at
import at.plot
import time
import h5py
import bisect

ring = at.load_mat('ALBA_II_20250129_divideSH1SV2SV4SV7_OCT_B.mat')


def FastTouschekTracking (ring, nvoltes=1000):
    st = time.time()
    momacc = np.empty((3,0)) # momentum acceptance
    poly = dopolyhedron(ring, nvoltes) # Polyhedron in s=0
    closedorbit = at.find_orbit6(ring, refpts=at.All)[1]
    opt = at.linopt6(ring)
    binvB = np.array([[1, 0], [opt[0]['alpha'][0], opt[0]['beta'][0]]]) 
    a = 0; b = 2**6 # low and high bounds.
    deltastep = 0.001
    
    for i in range(1,len(ring)):
        if ring[i].Length != 0: # for every relevant position in the ring
            shortring = ring[i:]
            closed = closedorbit[i]
            dpN = binarysearch(a, b, -deltastep, poly, shortring, closed, binvB) # conservative estimates
            dpP = binarysearch(a, b, deltastep, poly, shortring, closed, binvB)
            momacc = np.hstack((momacc, ([[i], [dpN], [dpP]]))) 
    temps = round((time.time()-st)/3600, 2)   

    plt.figure(figsize=(16,8))
    plt.plot(momacc[0], momacc[1], linestyle='-', color='blue')
    plt.plot(momacc[0], momacc[2], linestyle='-', color='blue')
    plt.title('Momentum Acceptance, FTT.  nturns='+str(nvoltes)+', time='+str(temps)+'h')
    plt.xlabel('element index')
    plt.ylabel('dp/p')
    plt.savefig('momentumacceptance_'+str(nvoltes)+'_.jpeg')

    name = 'momentumacceptance_'+str(nvoltes)+'.hdf5'
    f = h5py.File(name,'w')
    mp = f.create_group('momacc')
    mp.create_dataset('nturns', data=np.array([nvoltes]))
    mp.create_dataset('deltastep', data=np.array([deltastep]))
    mp.create_dataset('execution_time (h)', data=np.array([temps]))
    mp.create_dataset('dpP',data=momacc[2]) # Positive
    mp.create_dataset('dpN',data=momacc[1]) # Negative
    mp.create_dataset('indexs',data=momacc[0])
    spos_array = ring.get_s_pos(np.arange(0,len(ring)))
    mp.create_dataset('spos',data=spos_array[(momacc[0]).astype(np.int64)])
    f.close()

    return momacc


def isin(valor, conjunt):
     for i in range (0,len(conjunt)):
         if abs(conjunt[i] - valor) < 1e-6: # tolerància 1e-6
             return True, i
     return False, 0 
 

def dopolyhedron (ring, nvoltes): 
    st = time.time()
    step = 0.1 
    xvals = np.arange(-10, 10.1, step)
    pxvals = np.arange(1, -1.01, -step/10)
    punts = [(px,x) for px in pxvals for x in xvals]
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
    nangles = 180 
    angles = np.linspace(-np.pi, np.pi, nangles, endpoint=False)
    
    dpmax = 0.06
    dpResolution = 0.005
    
    pdext = find_dext_alt (ring, nvoltes, particules, dpmax, dpResolution, npart, nx)
    ndext = find_dext_alt (ring, nvoltes, particules, -dpmax, -dpResolution, npart, nx)
    ndeltes = int(1 + (pdext+abs(ndext))/0.005) # /0.01 per deltes cada 1%, /0.005 per deltes cada 0.5%
    deltes = np.linspace(ndext, pdext, ndeltes, endpoint=True)
    prepoly = np.zeros((ndeltes,nangles,3))
    for i in range(0,ndeltes):
        prepoly[i] = parfloodfillpx(ring, nvoltes, particules, npart, nx, binvB, nangles, angles, round(deltes[i],4)) 
       
    # IF using find_dext instead of find_dext_alt:    
    #pdext, pprepoly = find_dext(ring, nvoltes, particules, dpmax, dpResolution, npart, nx, binvB, nangles, angles)
    #ndext, nprepoly = find_dext(ring, nvoltes, particules, -dpmax, -dpResolution, npart, nx, binvB, nangles, angles)
    #ndeltes = int(1 + (pdext+abs(ndext))/0.005)
    #deltes = np.linspace(ndext, pdext, ndeltes, endpoint=True)
    #prepoly = np.zeros((ndeltes,nangles,3))
    #for i in range(0,ndeltes):
    #    inppre = isin(deltes[i], pprepoly[:,0,2])
    #    innpre = isin(deltes[i], nprepoly[:,0,2])
    #    if inppre[0]:
    #        prepoly[i] = pprepoly[inppre[1]]
    #    elif innpre[0]:
    #        prepoly[i] = nprepoly[innpre[1]]
    #    else:    
    #        prepoly[i] = parfloodfillpx(ring, nvoltes, particules, npart, nx, binvB, nangles, angles, round(deltes[i],4))

    nslices = int(1 + (pdext+abs(ndext))/0.001)
    slices = np.linspace(ndext, pdext, nslices, endpoint=True)
    polyhedron = interpolate(prepoly, deltes, nslices, slices, nangles)
    temps = round((time.time()-st)/3600, 2)
    
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    for i in range(0, nslices):
        ax.scatter(polyhedron[i][:,0]*1e3,polyhedron[i][:,1]*1e3,polyhedron[i][:,2]*1e2, s=5, c='dodgerblue')
    plt.title('DA Polyhedron, with Flood-Fill and interpolation.  nturns='+str(nvoltes)+', time='+str(temps)+'h')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('X\' [mm]'); ax.set_zlabel('$\delta$ $(\%)$')
    ax.set_xlim([-10, 10]); ax.set_ylim([-10, 10]); ax.set_zlim([ndext*1e2, pdext*1e2])
    ax.zaxis.labelpad=-1
    ax.view_init(elev=5, azim=-60, roll=0)
    plt.savefig('polyhedron_'+str(nvoltes)+'.jpeg')
    
    name = 'polyhedron_'+str(nvoltes)+'.hdf5'
    f = h5py.File(name,'w')
    mp = f.create_group('polyhedron')
    mp.create_dataset('nturns', data=np.array([nvoltes]))
    mp.create_dataset('ndeltas', data=np.array([ndeltes]))
    mp.create_dataset('positive_deltamax', data=np.array([pdext]))
    mp.create_dataset('negative_deltamax', data=np.array([ndext]))
    mp.create_dataset('nslices', data=np.array([nslices]))
    mp.create_dataset('nangles', data=np.array([nangles]))
    mp.create_dataset('execution_time (h)', data=np.array([temps]))
    mp.create_dataset('interpolated_polyhedron', data=polyhedron)
    mp.create_dataset('initial_polyhedron', data=prepoly)
    f.close()
    
    return polyhedron
 
    
def find_dext (ring, nvoltes, particules, dpmax, dpResolution, npart, nx, binvB, nangles, angles):
    ndp = int(dpmax/dpResolution)
    dp_values = np.linspace(dpResolution, dpmax, ndp, endpoint=True) 
    preprepoly = np.zeros((ndp,nangles,3)) 
    low = 0                                           
    high = len(dp_values)-1
    while high-low > 1: 
        m = int((high-low)/2)+low
        frontera = parfloodfillpx(ring, nvoltes, particules, npart, nx, binvB, nangles, angles, round(dp_values[m],4))
        if len(frontera) == 0:
            high = m
        else:
            low = m
            preprepoly[m] = frontera 
    for i in range(0, ndp):
        if preprepoly[i,0,2] == 0:
            preprepoly[i,0,2] = 1 
    return round(dp_values[low],4), preprepoly 


def find_dext_alt (ring, nvoltes, particules, dpmax, dpResolution, npart, nx):
    ndp = int(dpmax/dpResolution)
    dp_values = np.linspace(dpResolution, dpmax, ndp, endpoint=True)
    c = int(npart/2) # center
    idx = [c, c+1, c-1, c+nx, c-nx]
    low = 0                                           
    high = len(dp_values)-1
    while high-low > 1:
        m = int((high-low)/2)+low
        particules[4] = round(dp_values[m],4)
        center = np.array([particules[:,i] for i in idx]).T
        temp = (ring.track(center, nturns=nvoltes, losses=True, use_mp=True))[2]['loss_map']['islost']
        if (temp == False).all(): # si totes no estan perdudes
            low = m
        else:
            high = m
    return round(dp_values[low],4)
    
 
def parfloodfillpx (ring, nturns, particles, npart, nx, binvB, nangles, angles, delta):
    particles[4] = delta
    
    st = time.time()
    lost = np.empty((2,0)) # Lost particles (after tracking)
    notlost = np.empty((2,0)) # Not lost particles (after tracking)
    queue = [] # Particles to be tracked
    queue.extend([0,nx-1,npart-nx,npart-1]) # Start from the four corners and
    queue.extend([int(nx/2),int(npart/2)-int(nx/2),int(npart/2)+int(nx/2),int(npart-nx/2)]) # the corresponding midpoints
    done = [] # Keep count of the ones that have been tracked
    
    while len(queue) != 0:
        tracknext = np.empty((0,6)) # Particles that will be tracked in this iteration
        while len(tracknext) < 8: # 8 particles in every iteration,
            if len(queue) == 0: # unless the queue is empty
                break
            i = queue.pop(0) # Take the first particle out of the queue,
            if (0 <= i < npart) and (i not in done): # and if it hasn't been tracked yet,
                done.append(i)
                tracknext = np.vstack((tracknext, ([particles[:,i]]))) # add it to track
        temp = (ring.track(tracknext.T, nturns=nturns, losses=True, use_mp=True))[2]['loss_map']
        for l in range (0, len(temp)):
            i = done[-len(temp)+l]
            if temp['islost'][l]: # If the particle is lost,
                queue.append(i+1) # track the particle on its right
                queue.append(i-1) # left
                queue.append(i+nx) # bottom
                queue.append(i-nx) # and top
                lost = np.hstack((lost, ([[particles[0][i]], [particles[1][i]]])))
            else: # If the particle is not lost
                notlost = np.hstack((notlost, ([[particles[0][i]], [particles[1][i]]])))

    norm = binvB.dot(notlost) # Filtro partícules notlost amb radi mínim
    radis = np.sqrt((norm[0]-1e-5)**2 + norm[1]**2)
    thetas = np.arctan2(norm[1], norm[0]-1e-5)
    norm_lost = binvB.dot(lost)            
    radis_lost = np.sqrt((norm_lost[0]-1e-5)**2 + norm_lost[1]**2)
    thetas_lost = np.arctan2(norm_lost[1], norm_lost[0]-1e-5)
    marge = np.pi / nangles
    prefrontera = np.empty((0,3))
    for theta in angles:
        if theta == angles[0]:
            sector = (thetas >= angles[-1] + marge) | (thetas < theta + marge)
            sector_lost = (thetas_lost >= angles[-1] + marge) | (thetas_lost < theta + marge)
        else:
            sector = (thetas >= theta - marge) & (thetas < theta + marge)
            sector_lost = (thetas_lost >= theta - marge) & (thetas_lost < theta + marge)
        if np.any(sector):
            idx_candidats = np.where(sector)[0]
            idx_minim = idx_candidats[np.argmin(radis[idx_candidats])]
            radi_minim = radis[idx_minim]
            radis_sector_lost = radis_lost[np.where(sector_lost)[0]]
            if (radi_minim <= 1.1*radis_sector_lost).all(): 
                prefrontera = np.vstack((prefrontera, ([norm[0][idx_minim], norm[1][idx_minim], thetas[idx_minim]])))
    prefrontera = prefrontera[np.argsort(prefrontera[:,2])] 
    
    plt.figure(figsize=(8,8))
    plt.scatter(norm_lost[0]*1e3, norm_lost[1]*1e3, color='lightgray')
    plt.scatter(norm[0]*1e3, norm[1]*1e3, color='firebrick')
    plt.title('Dynamic Aperture, Flood-Fill.  nturns='+str(nturns)+', dp='+str(round(delta*1e2,3))+'%')
    plt.xlabel('x (mm)'); plt.ylabel('x\' (mm)')
    plt.xlim(-10,10); plt.ylim(-10,10) # de fet amb plt.ylim(-9.6,9.6) ja hi caben. quan ho faig es veuen irregulars les línies grises de dalt i baix!?
    plt.savefig('xpx_'+str(nturns)+'_'+str(round(delta*1e2,3))+'.jpeg')

    
    frontera = np.empty((0,3))
    # if len(prefrontera) < 30: # Only if using find_dext
    #    return frontera 
    
    # Col·loco partícules als angles que toquen. AIXÒ S'HA DE MILLORAR (?) (ells ho fan amb discretisePolygon i intersections)
    for theta in angles:
        idx = bisect.bisect_left(prefrontera[:,2], theta)
        if idx == 0: #or idx == len(prefrontera):            
            ang = [prefrontera[-1][2], prefrontera[0][2]+2*np.pi] # el segon angle és > pi but makes sense
            lam = (theta+2*np.pi - ang[0]) / (ang[1] - ang[0]) 
        elif idx == len(prefrontera): 
            ang = [prefrontera[-1][2], prefrontera[0][2]+2*np.pi] 
            lam = (theta - ang[0]) / (ang[1] - ang[0]) # theta ja és positiu, no cal fer-li res
            idx = 0
        else:
            ang = [prefrontera[idx-1][2], prefrontera[idx][2]]
            lam = (theta - ang[0]) / (ang[1] - ang[0])
        x = (1-lam)*prefrontera[idx-1][0] + lam*prefrontera[idx][0]
        xp = (1-lam)*prefrontera[idx-1][1] + lam*prefrontera[idx][1]
        frontera = np.vstack((frontera, ([x, xp, delta]))) 

    temps = round((time.time()-st)/60, 2)  
    plt.figure(figsize=(8,8))
    plt.scatter(frontera[:,0]*1e3, frontera[:,1]*1e3, color='firebrick')
    plt.title('Dynamic Aperture Border, Flood-Fill.  nturns='+str(nturns)+', dp='+str(round(delta*1e2,3))+'%, time='+str(temps)+'min')
    plt.xlabel('x (mm)'); plt.ylabel('x\' (mm)')
    plt.xlim(-10,10); plt.ylim(-10,10)
    plt.savefig('xpx_boundary_'+str(nturns)+'_'+str(round(delta*1e2,3))+'.jpeg')

    return frontera


def interpolate (prepoly, deltes, nslices, slices, nangles):
    polyhedron = np.zeros((nslices,nangles,3))

    for i in range(0,nslices): 
        done = isin(slices[i], deltes) # tot i que em sembla que aquí el in ja anava bé
        if done[0]: 
            polyhedron[i] = prepoly[done[1]]
        else:
            idx = bisect.bisect_left(deltes, slices[i])
            lam = (slices[i] - deltes[idx-1]) / (deltes[idx] - deltes[idx-1])
            polyhedron[i][:,0] = (1-lam)*prepoly[idx-1][:,0] + lam*prepoly[idx][:,0]
            polyhedron[i][:,1] = (1-lam)*prepoly[idx-1][:,1] + lam*prepoly[idx][:,1]
            polyhedron[i][:,2] = round(slices[i],4)
    
    return polyhedron


def binarysearch (a, b, deltastep, poly, shortring, closed, binvB):    
    while b-a > 1:
        m = (a+b)/2
        particula = np.array([closed[0]+1e-5, closed[1], closed[2]+1e-5, closed[3], closed[4]+m*deltastep, closed[5]])
        t = shortring.track(particula, nturns=1, losses=True)[0] # trasllat
        if not np.isnan(t[0][0][0][0]): # si no està perduda
            norm = binvB.dot(np.array([[t[0][0][0][0]], [t[1][0][0][0]]])) # normalitzem x, xp
            part = np.array([norm[0][0],norm[1][0],t[4][0][0][0]]) # només ens interessen x, xp i dp/p
            inside = particleinpoly(poly, part) 
            if inside: # == True
                a = m
            else:
                b = m
        else:
            b = m
    return a*deltastep


def particleinpoly (poly, part):
    nslices = len(poly) # això també podria posar-ho fora i no generar-ho cada vegada...
    slices = poly[:,0,2]
    nangles = len(poly[0])
    angles = np.linspace(-np.pi, np.pi, nangles, endpoint=False)

    if part[2] < slices[0]-5e-4 or part[2] > slices[-1]+5e-4: 
        return False 
    
    i = bisect.bisect_left(slices, part[2]) # avaluo la partícula al polígon amb dp/p més proper (en v.abs)
    if (i == nslices) or (i != 0 and part[2]-slices[i-1] < slices[i]-part[2]):
        i = i-1
        
    phi = np.arctan2(part[1], part[0])
    idx = bisect.bisect_left(angles, phi)
    if idx == nangles: # el cas 0 no cal pq és impossible q arctan retorni un angle més petit que -pi
        ant = idx-1; post = 0 
    else:
        ant = idx-1; post = idx
    varx = poly[i][post][0] - poly[i][ant][0]
    varxp = poly[i][post][1] - poly[i][ant][1]
    if part[1]*varx - part[0]*varxp > poly[i][ant][1]*varx - poly[i][ant][0]*varxp:
        return True
    return False


def touscheklifetime (ring, momacc):
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