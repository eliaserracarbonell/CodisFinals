"""FLOOD FILL algorithm"""

import time

import at
import at.plot
import matplotlib.pyplot as plt
import numpy as np


def floodfill(ring, nturns=1000, delta=0, xmax=10, ymax=5, xstep=0.1, ystep=0.1):
    r"""Finds the Dynamic Aperture of the lattice using Flood Fill.
    
    Parameters:
        ring:       Lattice definition
            
    Keyword Arguments:
        nturns:     Number of turns for the tracking. Default: 1000
        delta:      Momentum offset dp/p. Default: 0
        xmax:       Maximum value of x, in mm. Default: 10
        ymax:       Maximum value of y, in mm. Default: 5
        xstep:      Minumum distance between two particles in x axis, in mm. Default: 0.1
        ystep:      Minimum distance between two particles in y axis, in mm. Default: 0.1
        
    Returns:
        boundary:   (2,n) array: Coordinates of dynamic aperture boundary
        notlost:    (2,n) array: Coordinates of tracked particles that have survived
        lost:       (3,n) array: Coordinates of tracked particles that have not survived,
                                 and turn in which each got lost.
                                 
    Example:
        b, nl, l = floodfill(ring, nturns=500)                         
    """
    
    st = time.time()
    
    # Create the particle grid
    xvals = np.arange(-xmax, xmax+xstep, xstep)
    yvals = np.arange(ymax, -ymax-ystep, -ystep) 
    points = [(y,x) for y in yvals for x in xvals] # from left to right, from top to bottom
    nx = len(xvals)
    npart = len(points)
    coord = np.reshape(points, (npart,2))
    coord = np.transpose(coord)
    closed = at.find_orbit(ring)
    particles = np.zeros((6,npart))
    particles[0] = coord[1]*1e-3 + 1e-5
    particles[2] = coord[0]*1e-3 + 1e-5
    particles[4] = delta
    particles[5] = closed[0][5]
    
    # Track the particles for nturns using the flood-fill algorithm
    lost = np.empty((3,0)) # Lost particles after tracking
    notlost = np.empty((2,0)) # Not lost particles after tracking
    queue = [] # Particles to be tracked
    queue.append(0) # Start from the top left corner
    done = [] # Keep count of the particles that have been tracked
    while len(queue)!=0:
        i = queue.pop(0) # Take the first particle out of the queue and track it:
        if (0 <= i < npart) and (i not in done):
            done.append(i)
            temp = (ring.track(particles[:,i], nturns=nturns, losses=True))[2]['loss_map']
            if temp['islost'][0]: # If the particle is lost,
                queue.append(i+1) # track the particle on its right
                queue.append(i-1) # left
                queue.append(i+nx) # bottom
                queue.append(i-nx) # and top
                lost = np.hstack((lost, ([[particles[0][i]], [particles[2][i]], [temp['turn'][0]]])))
            else: # If the particle is not lost
                notlost = np.hstack((notlost, ([[particles[0][i]], [particles[2][i]]])))
    
    # Define the Dynamic Aperture boundary as the notlost particles closest to the center
    radiuses = np.sqrt((notlost[0]-1e-5)**2 + (2*(notlost[1]-1e-5))**2) # Radiuses of not lost particles
    thetas = np.arctan2(2*(notlost[1]-1e-5), notlost[0]-1e-5) # Angles of not lost particles
    radiuses_lost = np.sqrt((lost[0]-1e-5)**2 + (2*(lost[1]-1e-5))**2) # Radiuses of lost particles
    thetas_lost = np.arctan2(2*(lost[1]-1e-5), lost[0]-1e-5) # Angles of lost particles
    nangles = 180
    margin = np.pi / nangles
    angles = np.linspace(-np.pi, np.pi, nangles, endpoint=False)
    boundary = np.empty((2,0)) # Dynamic aperture
    for theta in angles:
        if theta == angles[0]: # Special case theta = -pi = pi
            sector = (thetas >= angles[-1] + margin) | (thetas < theta + margin)
            sector_lost = (thetas_lost >= angles[-1] + margin) | (thetas_lost < theta + margin)
        else: # Find the notlost particles in every sector
            sector = (thetas >= theta - margin) & (thetas < theta + margin)
            sector_lost = (thetas_lost >= theta - margin) & (thetas_lost < theta + margin)
        if np.any(sector): # and select the one with the minimum radius
            idx_candidates = np.where(sector)[0]
            idx_minimum = idx_candidates[np.argmin(radiuses[idx_candidates])] 
            minimum_radius = radiuses[idx_minimum]
            radiuses_sector_lost = radiuses_lost[np.where(sector_lost)[0]]
            if (minimum_radius <= 1.1*radiuses_sector_lost).all(): # to make sure it is not an outlier
                boundary = np.hstack((boundary, ([[notlost[0][idx_minimum]], [notlost[1][idx_minimum]]])))

    exec_time = round((time.time()-st)/60, 2)
    print('Execution time: '+str(exec_time)+' min')         
            
    return boundary, notlost, lost



def gradient_plot (boundary, lost, nturns):
    r"""Plots the Dynamic Aperture boundary found using Flood Fill,
    plus the particles that don't survive with a color gradient for the turn they got lost.
    Saves the result in a .jpeg file.
    
    Parameters:
        boundary:  (2,n) array: Coordinates of dynamic aperture boundary. (Result of floodfill)
        lost:      (3,n) array: Coordinates of lost particles and turn each got lost. (Result of floodfill)
        nturns:    Number of turns used for the floodfill tracking.
    
    Example:
        gradient_plot(b, l, nturns)
    """
    
    colors = np.where(lost[2] <= 1, 'black',
                     np.where(lost[2] <= 5, 'dimgrey',
                     np.where(lost[2] <= 25, 'darkgrey',
                     np.where(lost[2] <= 125, 'lightgrey', 'whitesmoke'))))
    colors_names = ['black', 'dimgrey', 'darkgrey', 'lightgrey', 'whitesmoke']
    labels = ['$\leq$1', '2-5', '6-25', '26-125', '126-'+str(nturns)]
    legend = [plt.Line2D([0], [0], marker='o', color='k', markerfacecolor=color, markersize=8, label=label)
              for color, label in zip(colors_names, labels)]
    
    plt.figure(figsize=(16,8))
    plt.scatter(lost[0]*1e3, lost[1]*1e3, c=colors)
    plt.plot(boundary[0]*1e3, boundary[1]*1e3, linestyle='-', linewidth=3, color='firebrick')
    plt.title('Dynamic Aperture found with Flood Fill, for nturns='+str(nturns)+'.')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.legend(handles=legend, title='turns until lost', loc='lower right')
    plt.savefig('FloodFill_DA_'+str(nturns)+'.jpeg')