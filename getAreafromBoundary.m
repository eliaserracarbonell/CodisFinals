function thearea = getAreafromBoundary(boundary)
% function thearea = getAreafromBoundary(boundary)
%  
%   boundary: (2,n) array.
%   thearea:  float.
%
% This function approximates the area inside a boundary.
% This algorithm is only valid for a function on the first quadrant.

% oblanco ALBA 2025mar04
[xx,ia] = sort(boundary(1,:));
yy = boundary(2,ia);

dx = diff(xx);
mask = dx > 0;
widthx = dx(mask);
heighty = yy(mask);

thearea = sum(widthx.*heighty);

end