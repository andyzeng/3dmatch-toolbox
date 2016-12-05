% ISCOLINEAR - are 3 points colinear
%
% Usage:  r = iscolinear(p1, p2, p3, flag)
%
% Arguments:
%        p1, p2, p3 - Points in 2D or 3D.
%        flag       - An optional parameter set to 'h' or 'homog'
%                     indicating that p1, p2, p3 are homogneeous
%                     coordinates with arbitrary scale.  If this is
%                     omitted it is assumed that the points are
%                     inhomogeneous, or that they are homogeneous with
%                     equal scale.
%
% Returns:
%        r = 1 if points are co-linear, 0 otherwise

% Copyright (c) 2004-2005 Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% http://www.csse.uwa.edu.au/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% February 2004
% January  2005 - modified to allow for homogeneous points of arbitrary
%                 scale (thanks to Michael Kirchhof)


function r = iscolinear(p1, p2, p3, flag)

    if nargin == 3   % Assume inhomogeneous coords
	flag = 'inhomog';
    end
    
    if ~all(size(p1)==size(p2)) | ~all(size(p1)==size(p3)) | ...
        ~(length(p1)==2 | length(p1)==3)                              
        error('points must have the same dimension of 2 or 3');
    end
    
    % If data is 2D, assume they are 2D inhomogeneous coords. Make them
    % homogeneous with scale 1.
    if length(p1) == 2    
        p1(3) = 1; p2(3) = 1; p3(3) = 1;
    end

    if flag(1) == 'h'
	% Apply test that allows for homogeneous coords with arbitrary
        % scale.  p1 X p2 generates a normal vector to plane defined by
        % origin, p1 and p2.  If the dot product of this normal with p3
        % is zero then p3 also lies in the plane, hence co-linear.
	r =  abs(dot(cross(p1, p2),p3)) < eps;
    else
	% Assume inhomogeneous coords, or homogeneous coords with equal
        % scale.
	r =  norm(cross(p2-p1, p3-p1)) < eps;
    end
    
