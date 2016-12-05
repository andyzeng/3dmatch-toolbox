% RANSACFITHOMOGRAPHY - fits 2D homography using RANSAC
%
% Usage:   [H, inliers] = ransacfithomography(x1, x2, t)
%
% Arguments:
%          x1  - 2xN or 3xN set of homogeneous points.  If the data is
%                2xN it is assumed the homogeneous scale factor is 1.
%          x2  - 2xN or 3xN set of homogeneous points such that x1<->x2.
%          t   - The distance threshold between data point and the model
%                used to decide whether a point is an inlier or not. 
%                Note that point coordinates are normalised to that their
%                mean distance from the origin is sqrt(2).  The value of
%                t should be set relative to this, say in the range 
%                0.001 - 0.01  
%
% Note that it is assumed that the matching of x1 and x2 are putative and it
% is expected that a percentage of matches will be wrong.
%
% Returns:
%          H       - The 3x3 homography such that x2 = H*x1.
%          inliers - An array of indices of the elements of x1, x2 that were
%                    the inliers for the best model.
%
% See Also: ransac, homography2d, homography1d

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

% February 2004 - original version
% July     2004 - error in denormalising corrected (thanks to Andrew Stein)
% August   2005 - homogdist2d modified to fit new ransac specification.

function [H, inliers] = ransacfithomography(x1, x2, t)

    if ~all(size(x1)==size(x2))
        error('Data sets x1 and x2 must have the same dimension');
    end
    
    [rows,npts] = size(x1);
    if rows~=2 & rows~=3
        error('x1 and x2 must have 2 or 3 rows');
    end
    
    if npts < 4
        error('Must have at least 4 points to fit homography');
    end
    
    if rows == 2    % Pad data with homogeneous scale factor of 1
        x1 = [x1; ones(1,npts)];
        x2 = [x2; ones(1,npts)];        
    end
        
    % Normalise each set of points so that the origin is at centroid and
    % mean distance from origin is sqrt(2).  normalise2dpts also ensures the
    % scale parameter is 1.  Note that 'homography2d' will also call
    % 'normalise2dpts' but the code in 'ransac' that calls the distance
    % function will not - so it is best that we normalise beforehand.
    [x1, T1] = normalise2dpts(x1);
    [x2, T2] = normalise2dpts(x2);
    
    s = 4;  % Minimum No of points needed to fit a homography.
    
    fittingfn = @homography2d;
    distfn    = @homogdist2d;
    degenfn   = @isdegenerate;
    % x1 and x2 are 'stacked' to create a 6xN array for ransac
    [H, inliers] = ransac([x1; x2], fittingfn, distfn, degenfn, s, t);
    
    % Now do a final least squares fit on the data points considered to
    % be inliers.
    H = homography2d(x1(:,inliers), x2(:,inliers));
    
    % Denormalise
    H = T2\H*T1;    

%----------------------------------------------------------------------
% Function to evaluate the symmetric transfer error of a homography with
% respect to a set of matched points as needed by RANSAC.

function [inliers, H] = homogdist2d(H, x, t);
    
    x1 = x(1:3,:);   % Extract x1 and x2 from x
    x2 = x(4:6,:);    
    
    % Calculate, in both directions, the transfered points    
    Hx1    = H*x1;
    invHx2 = H\x2;
    
    % Normalise so that the homogeneous scale parameter for all coordinates
    % is 1.
    
    x1     = hnormalise(x1);
    x2     = hnormalise(x2);     
    Hx1    = hnormalise(Hx1);
    invHx2 = hnormalise(invHx2); 
    
    d2 = sum((x1-invHx2).^2)  + sum((x2-Hx1).^2);
    inliers = find(abs(d2) < t);    
    
    
%----------------------------------------------------------------------
% Function to determine if a set of 4 pairs of matched  points give rise
% to a degeneracy in the calculation of a homography as needed by RANSAC.
% This involves testing whether any 3 of the 4 points in each set is
% colinear. 
     
function r = isdegenerate(x)

    x1 = x(1:3,:);    % Extract x1 and x2 from x
    x2 = x(4:6,:);    
    
    r = ...
    iscolinear(x1(:,1),x1(:,2),x1(:,3)) | ...
    iscolinear(x1(:,1),x1(:,2),x1(:,4)) | ...
    iscolinear(x1(:,1),x1(:,3),x1(:,4)) | ...
    iscolinear(x1(:,2),x1(:,3),x1(:,4)) | ...
    iscolinear(x2(:,1),x2(:,2),x2(:,3)) | ...
    iscolinear(x2(:,1),x2(:,2),x2(:,4)) | ...
    iscolinear(x2(:,1),x2(:,3),x2(:,4)) | ...
    iscolinear(x2(:,2),x2(:,3),x2(:,4));
    
