% MATCHBYCORRELATION - match image feature points by correlation
%
% Function generates putative matches between previously detected
% feature points in two images by looking for points that are maximally
% correlated with each other within windows surrounding each point.
% Only points that correlate most strongly with each other in *both*
% directions are returned.
% This is a simple-minded N^2 comparison.
%
% Usage: [m1, m2, p1ind, p2ind, cormat] = ...
%                 matchbycorrelation(im1, p1, im2, p2, w, dmax)
%
% Arguments:
%         im1, im2 - Images containing points that we wish to match.
%         p1, p2   - Coordinates of feature pointed detected in im1 and
%                    im2 respectively using a corner detector (say Harris
%                    or phasecong2).  p1 and p2 are [2xnpts] arrays though
%                    p1 and p2 are not expected to have the same number
%                    of points.  The first row of p1 and p2 gives the row
%                    coordinate of each feature point, the second row
%                    gives the column of each point.
%         w        - Window size (in pixels) over which the correlation
%                    around each feature point is performed.  This should
%                    be an odd number.
%         dmax     - (Optional) Maximum search radius for matching
%                    points.  Used to improve speed when there is little
%                    disparity between images. Even setting it to a generous
%                    value of 1/4 of the image size gives a useful
%                    speedup. If this parameter is omitted it defaults to Inf. 
%
%
% Returns:
%         m1, m2   - Coordinates of points selected from p1 and p2
%                    respectively such that (putatively) m1(:,i) matches
%                    m2(:,i). m1 and m2 are [2xnpts] arrays defining the
%                    points in each of the images in the form [row;col].
%   p1ind, p2ind   - Indices of points in p1 and p2 that form a match.  Thus,
%                    m1 = p1(:,p1ind) and m2 = p2(:,p2ind)
%         cormat   - Correlation matrix; rows correspond to points in p1,
%                    columns correspond to points in p2

% Copyright (c) 2004-2009 Peter Kovesi
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

% February 2004    - Original version
% May      2004    - Speed improvements + constraint on search radius for
%                    additional speed
% August   2004    - Vectorized distance calculation for more speed
%                    (thanks to Daniel Wedge)
% December 2009    - Added return of indices of matching points from original
%                    point arrays

function [m1, m2, p1ind, p2ind, cormat] = ...
                  matchbycorrelation(im1, p1, im2, p2, w, dmax)

if nargin == 5
    dmax = Inf;
end

im1 = double(im1);
im2 = double(im2);

% Subtract image smoothed with an averaging filter of size wXw from
% each of the images.  This compensates for brightness differences in
% each image.  Doing it now allows faster correlation calculation.
im1 = im1 - filter2(fspecial('average',w),im1);
im2 = im2 - filter2(fspecial('average',w),im2);    

% Generate correlation matrix
cormat = correlatiomatrix(im1, p1, im2, p2, w, dmax);
[corrows,corcols] = size(cormat);

% Find max along rows give strongest match in p2 for each p1
[mp2forp1, colp2forp1] = max(cormat,[],2);

% Find max down cols give strongest match in p1 for each p2    
[mp1forp2, rowp1forp2] = max(cormat,[],1);    

% Now find matches that were consistent in both directions
p1ind = zeros(1,length(p1));  % Arrays for storing matched indices
p2ind = zeros(1,length(p2));    
indcount = 0;    
for n = 1:corrows
    if rowp1forp2(colp2forp1(n)) == n  % consistent both ways
        indcount = indcount + 1;
        p1ind(indcount) = n;
        p2ind(indcount) = colp2forp1(n);
    end
end

% Trim arrays of indices of matched points
p1ind = p1ind(1:indcount);    
p2ind = p2ind(1:indcount);        

% Extract matched points from original arrays
m1 = p1(:,p1ind);  
m2 = p2(:,p2ind);    


%-------------------------------------------------------------------------    
% Function that does the work.  This function builds a correlation matrix
% that holds the correlation strength of every point relative to every
% other point.  While this seems a bit wasteful we need all this data if
% we want to find pairs of points that correlate maximally in both
% directions.
%
% This code assumes im1 and im2 have zero mean.  This speeds the
% calculation of the normalised correlation measure.

function cormat = correlatiomatrix(im1, p1, im2, p2, w, dmax)

if mod(w, 2) == 0
    error('Window size should be odd');
end

[rows1, npts1] = size(p1);
[rows2, npts2] = size(p2);    

% Initialize correlation matrix values to -infinty
cormat = -ones(npts1,npts2)*Inf;

if rows1 ~= 2 | rows2 ~= 2
    error('Feature points must be specified in 2xN arrays');
end

[im1rows, im1cols] = size(im1);
[im2rows, im2cols] = size(im2);    

r = (w-1)/2;   % 'radius' of correlation window

% For every feature point in the first image extract a window of data
% and correlate with a window corresponding to every feature point in
% the other image.  Any feature point less than distance 'r' from the
% boundary of an image is not considered.

% Find indices of points that are distance 'r' or greater from
% boundary on image1 and image2;
n1ind = find(p1(1,:)>r & p1(1,:)<im1rows+1-r & ...
    p1(2,:)>r & p1(2,:)<im1cols+1-r);

n2ind = find(p2(1,:)>r & p2(1,:)<im2rows+1-r & ...
    p2(2,:)>r & p2(2,:)<im2cols+1-r);    

for n1 = n1ind            
    % Generate window in 1st image   	
    w1 = im1(p1(1,n1)-r:p1(1,n1)+r, p1(2,n1)-r:p1(2,n1)+r);
    % Pre-normalise w1 to a unit vector.
    w1 = w1./sqrt(sum(sum(w1.*w1)));

    % Identify the indices of points in p2 that we need to consider.
    if dmax == inf
	n2indmod = n2ind; % We have to consider all of n2ind
    
    else     % Compute distances from p1(:,n1) to all available p2.
	p1pad = repmat(p1(:,n1),1,length(n2ind));
	dists2 = sum((p1pad-p2(:,n2ind)).^2);
	% Find indices of points in p2 that are within distance dmax of
        % p1(:,n1) 
	n2indmod = n2ind(find(dists2 < dmax^2)); 
    end

    % Calculate noralised correlation measure.  Note this gives
    % significantly better matches than the unnormalised one.

    for n2 = n2indmod 
        % Generate window in 2nd image
        w2 = im2(p2(1,n2)-r:p2(1,n2)+r, p2(2,n2)-r:p2(2,n2)+r);
        cormat(n1,n2) = sum(sum(w1.*w2))/sqrt(sum(sum(w2.*w2)));
    end
end
