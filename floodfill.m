function [ boundary, pnotlostdata, plostdata] = floodfill(ring, varargin)
% [ boundary, pnotlostdata, plostdata] = floodfill(ring)
%
% Finds the D.A. of the lattice using Flood Fill.
%
% Flood fill tracks particles from the exterior to the border of the D.A.
% The lost particles are returned in plostdata.
% The not lost particles are returned in pnotlostdata.
% The boundary is found by choosing a center and spliting the space in n
% sectors. Then, the particle with minimum distance to the center per
% sector returned as part of the boundary.
% 
% Parameters:
%   ring:       AT lattice.
% 
% Keyword Arguments:
%   nturns:     Number of turns for the tracking. Default: 1000
%   window:     Min and max coordinate range.
%               Default [-10e-3,10e-3,-5e-3,5e-3]
%   delta:      Momentum offset dp/p. Default: 0
%   gridsize:   Number of steps per axis. Default [10,10]
%   axes:       Indexes of axes to be scanned. Default [1,3]
%   sixdoffset: Offset to be added. Default zeros(6,1)
%   userefcenter: Use the 'refcenter' as the boundary center. Default 0.
%   refcenter:  Used only when 'userefcenter' is not 0.
%               Define the center of the boundary (6,1).
%               If not given the mean coordinate of the surviving
%               particles is calculated per axis.
%   verbose:    Print extra info. Default 0.
%   docloseorbit: Calculate the closed orbit. Default 1.
%   closedorbit: Only valid if docloseorbit is 0.
%               Defines the closed orbit (6,1). Default zeros(6,1).
%   centeronorbit: Add the closed orbit to the tracked particles.
%               Default 1.
%   nangles:    Split the boundary on n equal sectors. Default 180.
%   parallel:   Not implemented yet.
%   epsilon_offset: Small deviation to add to the tracked coordinates.
%               Default [10e-5 10e-5].
% 
% Returns:
%     boundary: (2,n) array: Coordinates of dynamic aperture boundary
%     notlost:  (2,n) array: Initial coordinates of tracked particles that
%               have survived.
%     lost:     (3,n) array: Initial coordinates of tracked particles that
%               have not survived, and turn in which each got lost.
% 
% Example:
%     [b, nl, l] = floodfill(THERING, nturns=500)

% Author : E. Serra,  UAB and ALBA,  2025 original version in python
% Edited : O. Blanco, ALBA,          2025 matlab version

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional arguments
    p = inputParser;
    addOptional(p,'nTurns',1000);
    addOptional(p,'window',[-10e-3,10e-3,-5e-3,5e-3]);
    addOptional(p,'delta',0);
    addOptional(p,'gridsize',[10,10]);
    addOptional(p,'axes',[1,3]);
    addOptional(p,'sixdoffset',zeros(6,1));
    addOptional(p,'userefcenter',0);
    addOptional(p,'refcenter',zeros(6,1));
    addOptional(p,'verbose',0);
    addOptional(p,'docloseorbit',1);
    addOptional(p,'closeorbit',zeros(6,1));
    addOptional(p,'centeronorbit',1);
    addOptional(p,'nangles',180);
    addOptional(p,'parallel',false);
    addOptional(p,'epsilon_offset',[1e-5 1e-5]);
    parse(p,varargin{:});
    par = p.Results;

    %% Initialize variables
    nturns = par.nTurns;
    window = par.window;
    delta = par.delta;
    gridsize = par.gridsize;
    sixdoffset = par.sixdoffset;
    userefcenter = par.userefcenter;
    refcenter = par.refcenter;
    verbose = par.verbose;
    docloseorbit = par.docloseorbit;
    closeorbit = par.closeorbit;
    centeronorbit = par.centeronorbit;
    axes = par.axes;
    nangles = par.nangles;
    parallel = par.parallel;
    epsilon_offset = par.epsilon_offset;

    if verbose
        fprintf('Flood fill starts.\n');
    end
   
    %% Create the particle grid
    xvals = linspace(window(1),window(2),gridsize(1));
    yvals = linspace(window(3),window(4),gridsize(2));
    nx = length(xvals);
    ny = length(yvals);
    npart = nx*ny;
    points = zeros(2,npart);
    if verbose
        fprintf('Points to be checked %d.\n',npart);
    end
    ii = 1;
    for ix = 1:nx
        for iy = 1:ny
            points(:,ii) = [xvals(ix),yvals(iy)]';
            ii = ii + 1;
        end
    end
    if docloseorbit
        if verbose
            fprintf("Calculating the closed orbit.\n");
        end
        if check_6d(ring)
            if verbose
                fprintf('Ring is 6D.\n')
            end
            closed = findorbit6(ring);
        else
            if verbose
                fprintf('Ring is 4D.\n')
            end
            closed = [findorbit4(ring); 0 ; 0];
        end
    else
        if verbose
            fprintf('Using the given closed orbit.\n')
        end
        closed = closeorbit;
    end
    if centeronorbit
        if verbose
            fprintf('Centering on orbit.\n');
        end
        sixdoffset = sixdoffset + closed;
    end
    particles = zeros(6,npart);
    particles = particles + sixdoffset;
    particles(axes(1),:) = particles(axes(1),:) + points(1,:);
    particles(axes(2),:) = particles(axes(2),:) + points(2,:);
    particles(1,:) = particles(1,:)  + epsilon_offset(1);
    particles(3,:) = particles(3,:)  + epsilon_offset(2);
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

    if verbose
        fprintf('Tracking...\n');
    end

    while ~isempty(thequeue)
        i = thequeue(end); % Take the first particle out of the queue and track it:
        thequeue(end) = [];
        if (1 <= i && i <= npart) && ~ismember(i,idxpartdone)
            idxpartdone(end+1) = i;
            [~, ~, ~, lossinfo] = ringpass(ring, particles(:,i), nturns);
            if lossinfo.lost % If the particle is lost,
                masknext =  [...
                            ~ismember(i+1,idxpartdone), ...
                            ~ismember(i-1,idxpartdone), ...
                            ~ismember(i+nx,idxpartdone), ...
                            ~ismember(i-nx,idxpartdone), ...
                            ];
                nextpoints = nonzeros(masknext .* [i+1 i-1 i+nx i-nx])';
                thequeue= [thequeue nextpoints];
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

    if verbose
        fprintf('Tracking done.\n');
    end

    % debug
    % figure; scatter(pnotlostdata(1,:),pnotlostdata(2,:))
    % figure; scatter(plostdata(1,:),plostdata(2,:))
    
    %% Define the Dynamic Aperture boundary as the notlost particles closest to the center
    % Define distance to reference
    if userefcenter
        if verbose
            fprintf('Using the defined center.\n');
        end
        refc = refcenter;
    else
        if verbose
            fprintf('Calculating the barycenter.\n');
        end
        refc = zeros(6,1);
        refc(axes(1)) = mean(pnotlostdata(1,:));
        refc(axes(2)) = mean(pnotlostdata(2,:));
    end
    thecenter2d = definecenter(refc, axes, epsilon_offset);
    [radii, thetas] = calcdistance(pnotlostdata,thecenter2d);
    [radii_lost, thetas_lost] = calcdistance(plostdata,thecenter2d);

    % debug
    % figure; polarplot(thetas,radii,'o')
    % figure; polarplot(thetas_lost,radii_lost,'o')

    if verbose
        fprintf('Finding the boundary sector by sector.\n');
    end
    % Divide the circle in sectors and check if particles survive.
    % Then, choose the one with lowest amplitude.
    anglemin = -pi;
    anglemax = pi;
    halftheangle = (anglemax-anglemin) / (2*nangles);
    angles = linspace(anglemin, anglemax, nangles);
    angles = angles(1:end-1);
    % Dynamic aperture
    boundary = [];
    for theangle = angles % interval [-pi,pi)
        % find the notlost particles in every sector
        if theangle == angles(1)
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
        % and select the one with the minimum radius per sector
        if any(sector)
            idx_candidates = find(sector);
            [minimum_radius, idx_minimum_candidate] = min(radii(sector));
            idx_minimum_rad = idx_candidates(idx_minimum_candidate);
            radii_sector_lost = radii_lost(sector_lost);
            if ~any(sector_lost) || ...
                    all((minimum_radius <= 1.1*radii_sector_lost))
                boundary(:,end+1) = [ ...
                                        pnotlostdata(1,idx_minimum_rad), ...
                                        pnotlostdata(2,idx_minimum_rad) ...
                                    ];
            end
        end
    end

    if verbose
        fprintf('Flood fill has finished.\n');
    end
end

function the2dcenter = definecenter(refcenter,axes,offset)
%  the2dcenter = definecenter(refcenter,axes,offset)
%
%  Define the center coordinates [x1,x2] from the 6d refcenter,
%  the axes, and the offset for tracking
    x1 = refcenter(axes(1)) + (axes(1) == 1)*offset(1);
    x2 = refcenter(axes(2)) + (axes(1) == 3)*offset(2);
    the2dcenter = [x1; x2];
end

function [radii,thetas] = calcdistance(twod_points,thecenter)
%  [radii,thetas] = calcdistance(twod_points,thecenter)
%
%  Calculate the radii and angles for a number of 2D points and a center
    if ~isempty(twod_points)
        distax1 = twod_points(1,:) - thecenter(1);
        distax2 = twod_points(2,:) - thecenter(2);
        radii = sqrt( distax1.^2 + distax2.^2);
        thetas = atan2( distax2, distax1);
    else
        radii = [];
        thetas = [];
    end
end
