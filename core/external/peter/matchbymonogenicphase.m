% MATCHBYMONOGENICPHASE - match image feature points using monogenic phase data
%
% Function generates putative matches between previously detected
% feature points in two images by looking for points that have minimal
% differences in monogenic phase data within windows surrounding each point.
% Only points that correlate most strongly with each other in *both*
% directions are returned.  This is a simple-minded N^2 comparison.
%
% This matcher performs rather well relative to normalised greyscale
% correlation.  Typically there are more putative matches found and fewer
% outliers.  There is a greater computational cost in the pre-filtering stage
% but potentially the matching stage is much faster as each pixel is effectively
% encoded with only 3 bits. (Though this potential speed is not realized in this
% implementation)
%
% Usage: [m1,m2] = matchbymonogenicphase(im1, p1, im2, p2, w, dmax, ...
%                                   nscale, minWaveLength, mult, sigmaOnf)
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
%         w        - Window size (in pixels) over which the phase bit codes
%                    around each feature point are matched.  This should
%                    be an odd number.
%         dmax     - Maximum search radius for matching points.  Used to 
%                    improve speed when there is little disparity between
%                    images.  Even setting it to a generous value of 1/4 of
%                    the image size gives a useful speedup. 
%         nscale   - Number of filter scales.
%         minWaveLength - Wavelength of smallest scale filter.
%         mult     - Scaling factor between successive filters.
%         sigmaOnf - Ratio of the standard deviation of the Gaussian
%                    describing the log Gabor filter's transfer function in
%                    the frequency domain to the filter center frequency. 
%
%
% Returns:
%         m1, m2   - Coordinates of points selected from p1 and p2
%                    respectively such that (putatively) m1(:,i) matches
%                    m2(:,i). m1 and m2 are [2xnpts] arrays defining the
%                    points in each of the images in the form [row;col].
%
%
% I have had good success with the folowing parameters:
%
%    w = 11;         Window size for correlation matching, 7 or greater
%                    seems fine.
%    dmax = 50; 
%    nscale = 1;     Just one scale can give very good results. Adding
%                    another scale doubles computation 
%    minWaveLength = 10;
%    mult = 4;       This is irrelevant if only one scale is used.  If you do
%                    use more than one scale try values in the range 2-4.
%    sigmaOnf = .2;  This results in a *very* large bandwidth filter.  A
%                    large bandwidth seems to be very important in the
%                    matching performance.
%
% See Also:  MATCHBYCORRELATION, MONOFILT

% Copyright (c) 2005 Peter Kovesi
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

% May 2005    - Original version adapted from matchbycorrelation.m


function [m1,m2,cormat] = matchbymonogenicphase(im1, p1, im2, p2, w, dmax, ...
                            nscale, minWaveLength, mult, sigmaOnf)

    orientWrap = 0;
    [f1, h1f1, h2f1, A1] = ...
        monofilt(im1, nscale, minWaveLength, mult, sigmaOnf, orientWrap);

    [f2, h1f2, h2f2, A2] = ...
        monofilt(im2, nscale, minWaveLength, mult, sigmaOnf, orientWrap);

    % Normalise filter outputs to unit vectors (should also have masking for
    % unreliable filter outputs)
    for s = 1:nscale
%       f1{s} = f1{s}./A1{s}; f2{s} = f2{s}./A2{s};
%       h1f1{s} = h1f1{s}./A1{s}; h1f2{s} = h1f2{s}./A2{s};     
%       h2f1{s} = h2f1{s}./A1{s}; h2f2{s} = h2f2{s}./A2{s};             
        
        % Try quantizing oriented phase vector to 8 octants to see what
        % effect this has (Performance seems to be reduced only slightly)
        f1{s} = sign(f1{s}); f2{s} = sign(f2{s}); 
        h1f1{s} = sign(h1f1{s}); h1f2{s} = sign(h1f2{s});               
        h2f1{s} = sign(h2f1{s}); h2f2{s} = sign(h2f2{s});                       
    end
    
    % Generate correlation matrix
    cormat = correlationmatrix(f1, h1f1, h2f1, p1, ...
                               f2, h1f2, h2f2, p2, w, dmax);

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
% Function that does the work.  This function builds a 'correlation' matrix
% that holds the correlation strength of every point relative to every other
% point.  While this seems a bit wasteful we need all this data if we want
% to find pairs of points that correlate maximally in both directions.

function cormat = correlationmatrix(f1, h1f1, h2f1, p1, ...
                                    f2, h1f2, h2f2, p2, w, dmax)
    
    if mod(w, 2) == 0 | w < 3
        error('Window size should be odd and >= 3');
    end

    r = (w-1)/2;   % 'radius' of correlation window
    
    [rows1, npts1] = size(p1);
    [rows2, npts2] = size(p2);    
    
    if rows1 ~= 2 | rows2 ~= 2
        error('Feature points must be specified in 2xN arrays');
    end    
    
    % Reorganize monogenic phase data into a 4D matrices for convenience

    [im1rows,im1cols] = size(f1{1});
    [im2rows,im2cols] = size(f2{1});
    nscale = length(f1);    
    phase1 = zeros(im1rows,im1cols,nscale,3);
    phase2 = zeros(im2rows,im2cols,nscale,3);    
   
    for s = 1:nscale
        phase1(:,:,s,1) = f1{s}; phase1(:,:,s,2) = h1f1{s}; phase1(:,:,s,3) = h2f1{s};
        phase2(:,:,s,1) = f2{s}; phase2(:,:,s,2) = h1f2{s}; phase2(:,:,s,3) = h2f2{s};    
    end

    % Initialize correlation matrix values to -infinity
    cormat = repmat(-inf, npts1, npts2);
        
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
        
        % Generate window in 1st image          
        w1 = phase1(p1(1,n1)-r:p1(1,n1)+r, p1(2,n1)-r:p1(2,n1)+r, :, :);

        for n2 = n2indmod 
            % Generate window in 2nd image
            w2 = phase2(p2(1,n2)-r:p2(1,n2)+r, p2(2,n2)-r:p2(2,n2)+r, :, :);
            % Compute dot product as correlation measure 
            cormat(n1,n2) = w1(:)'*w2(:);
            
            %   *** Need to add  mask stuff
        end
    end
    