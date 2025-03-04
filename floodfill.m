function [ boundary, pnotlostdata, plostdata] = floodfill(ring, varargin)
% floodfill(ring)
%
% Finds the Dynamic Aperture of the lattice using Flood Fill.
% 
% Parameters:
%     ring:       Lattice definition
% 
% Keyword Arguments:
%     nturns:     Number of turns for the tracking. Default: 1000
%     delta:      Momentum offset dp/p. Default: 0
%     xmax:       Maximum value of x, in mm. Default: 10
%     ymax:       Maximum value of y, in mm. Default: 5
%     xstep:      Minumum distance between two particles in x axis, in mm. Default: 0.1
%     ystep:      Minimum distance between two particles in y axis, in mm. Default: 0.1
% 
% Returns:
%     boundary:   (2,n) array: Coordinates of dynamic aperture boundary
%     notlost:    (2,n) array: Coordinates of tracked particles that have survived
%     lost:       (3,n) array: Coordinates of tracked particles that have not survived,
%                              and turn in which each got lost.
% 
% Example:
%     [b, nl, l] = floodfill(THERING, nturns=500)                         

% Author : E. Serra,  ALBA,  2025 original version in python
% Edited : O. Blanco, ALBA,  2025 matlab version

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional arguments
    p = inputParser;
    addOptional(p,'nTurns',1000);
    addOptional(p,'window',[-10e-3,10e-3,-5e-3,5e-3]);
    addOptional(p,'delta',0);
    addOptional(p,'gridsize',[10,10]);
    addOptional(p,'sixdoffset',zeros(6,1));
    addOptional(p,'refcenter',zeros(6,1));
    addOptional(p,'verbose',0);
    addOptional(p,'docloseorbit',1);
    addOptional(p,'closeorbit',zeros(6,1));
    addOptional(p,'centeronorbit',1);
    addOptional(p,'axes',[1,3]);
    addOptional(p,'nangles',180);
    addOptional(p,'parallel',false);
    addOptional(p,'epsilon_offset',1e-5);
    parse(p,varargin{:});
    par = p.Results;

    %% Initialize variables
    nturns = par.nTurns;
    window = par.window;
    delta = par.delta;
    gridsize = par.gridsize;
    sixdoffset = par.sixdoffset;
    refcenter = par.refcenter;
    verbose = par.verbose;
    docloseorbit = par.docloseorbit;
    closeorbit = par.closeorbit;
    centeronorbit = par.centeronorbit;
    axes = par.axes;
    nangles = par.nangles;
    parallel = par.parallel;
    epsilon_offset = par.epsilon_offset;
   
    %% Create the particle grid
    xvals = linspace(window(1),window(2),gridsize(1));
    yvals = linspace(window(3),window(4),gridsize(2));
    nx = length(xvals);
    ny = length(yvals);
    npart = nx*ny;
    points = zeros(2,npart);
    ii = 1;
    for ix = 1:nx
        for iy = 1:ny
            points(:,ii) = [xvals(ix),yvals(iy)]';
            ii = ii + 1;
        end
    end
    if docloseorbit
        if check_6d(ring)
            closed = findorbit6(ring);
        else
            closed = [findorbit4(ring); 0 ; 0];
        end
    else
        closed = closeorbit;
    end
    if centeronorbit 
        sixdoffset = sixdoffset + closed;
    end
    particles = zeros(6,npart);
    particles = particles + sixdoffset;
    particles(axes(1),:) = particles(axes(1),:) + points(1,:);
    particles(axes(2),:) = particles(axes(2),:) + points(2,:);
    particles(1,:) = particles(1,:)  + epsilon_offset;
    particles(3,:) = particles(3,:)  + epsilon_offset;
    particles(5,:) = particles(5,:)  + delta;

    %% Track the particles for nturns using the flood-fill algorithm
    plostdata = []; % Lost particles after tracking
    pnotlostdata = []; % Not lost particles after tracking
    thequeue = []; % Particles to be tracked
    thequeue(1:nx) = (0:(nx-1))*ny+1; % top border
    thequeue(nx+1:nx+ny-1) = 2:ny; % left side
    thequeue(nx+ny:nx+ny+ny-2) = ((nx-1)*ny+2):(nx*ny); % right side
    thequeue(nx+ny+ny-1:nx+ny+ny+nx-4) = (1:(nx-2))*ny+ny; % bottom side
    idxpartdone = []; % Keep count of the particles that have been tracked
    while ~isempty(thequeue)
        i = thequeue(end); % Take the first particle out of the queue and track it:
        thequeue(end) = [];
        if (1 <= i && i <= npart) && ~ismember(i,idxpartdone)
            idxpartdone(end+1) = i;
            [~, ~, ~, lossinfo] = ringpass(ring, particles(:,i), nturns);
            if lossinfo.lost % If the particle is lost,
                thequeue(end+1) = i+1;  % track the particle on its right
                thequeue(end+1) = i-1;  % left
                thequeue(end+1) = i+nx; % bottom
                thequeue(end+1) = i-nx; % and top
                plostdata(:,end+1) = [ ...
                                        particles(axes(1),i); ...
                                        particles(axes(2),i); ...
                                        lossinfo.turn ...
                                     ];
            else % If the particle is not lost
                pnotlostdata(:,end+1) = [ ...
                                        particles(axes(1),i); ...
                                        particles(axes(2),i); ...
                                        ];
            end
        end
    end

    % debug
    % figure; scatter(pnotlostdata(1,:),pnotlostdata(2,:))
    % figure; scatter(plostdata(1,:),plostdata(2,:))
    
    %% Define the Dynamic Aperture boundary as the notlost particles closest to the center
    % Define distance to reference
    thecenter2d = definecenter(refcenter, axes, epsilon_offset);
    [radii, thetas] = calcdistance(pnotlostdata,thecenter2d);
    [radii_lost, thetas_lost] = calcdistance(plostdata,thecenter2d);

    % debug
    % figure; polarplot(thetas,radii,'o')
    % figure; polarplot(thetas_lost,radii_lost,'o')

    quadrants = false(1,4);
    if window(2) > refcenter(1) && window(4) > refcenter(3)
        quadrants(1) = true;
    end
    if window(1) < refcenter(1) && window(4) > refcenter(3)
        quadrants(2) = true;
    end
    if window(1) < refcenter(1) && window(3) < refcenter(3)
        quadrants(3) = true;
    end
    if window(2) > refcenter(1) && window(3) < refcenter(3)
        quadrants(4) = true;
    end
   
    % angle range is defined by  the quadrants
    shifted_qu = circshift(quadrants,2);
    anglerange = (pi/2*(0:3)-pi+pi/4) .* shifted_qu;
    anglemin =min(anglerange(shifted_qu))-pi/4;
    anglemax =max(anglerange(shifted_qu))+pi/4;

    halftheangle = (anglemax-anglemin) / (2*nangles);
    angles = linspace(anglemin, anglemax, nangles);
    angles = angles(1:end-1);
    % Dynamic aperture
    boundary = [];
    for theangle = angles % interval [-pi,pi)
        % find the notlost particles in every sector
        if theangle == angles(1) && quadrants(3) && quadrants(2)
            % Special case theta = -pi = pi
            sector =        (thetas      >= (angles(end-1) + halftheangle)) ...
                          | (thetas      <  (theangle + halftheangle));
            sector_lost =   (thetas_lost >= (angles(end-1) + halftheangle)) ...
                          | (thetas_lost <  (theangle + halftheangle));
        else
            sector =        (thetas >= (theangle - halftheangle)) ...
                          & (thetas <  (theangle + halftheangle));
            sector_lost =   (thetas_lost >= (theangle - halftheangle)) ...
                          & (thetas_lost <  (theangle + halftheangle));
        end
        % and select the one with the minimum radius
        if any(sector) && any(sector_lost)
            %fprintf('%.3f\n',theangle);
            idx_tmp = find(sector);
            idx_candidates = idx_tmp(1);
            [~, idx_tmp] = min(radii(idx_candidates));
            idx_minimum = idx_candidates(idx_tmp); 
            minimum_radius = radii(idx_minimum);
            idx_tmp = find(sector_lost);
            radiuses_sector_lost = radii_lost(idx_tmp(1));
            if all((minimum_radius <= 1.1*radiuses_sector_lost))
                % to make sure it is not an outlier
                boundary(:,end+1) = [ ...
                                        pnotlostdata(1,idx_minimum), ...
                                        pnotlostdata(2,idx_minimum) ...
                                    ];
            end
        end
    end
end

function the2dcenter = definecenter(refcenter,axes,offset)
%  the2dcenter = definecenter(refcenter,axes,offset)
%
%  Define the center coordinates [x1,x2] from the 6d refcenter,
%  the axes, and the offset for tracking
    x1 = refcenter(axes(1)) + (axes(1) == 1)*offset;
    x2 = refcenter(axes(2)) + (axes(1) == 3)*offset;
    the2dcenter = [x1; x2];
end

function [radii,thetas] = calcdistance(twod_points,thecenter)
%  [radii,thetas] = calcdistance(twod_points,thecenter)
%
%  Calculate the radii and angles for a number of 2D points and a center
    distax1 = twod_points(1,:) - thecenter(1);
    distax2 = twod_points(2,:) - thecenter(2);
    radii = sqrt( distax1.^2 + distax2.^2);
    thetas = atan2( distax2, distax1);
end