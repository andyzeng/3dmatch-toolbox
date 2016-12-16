% HARRIS - Harris corner detector
%
% Usage:                 cim = harris(im, sigma)
%                [cim, r, c] = harris(im, sigma, thresh, radius, disp)
%  [cim, r, c, rsubp, csubp] = harris(im, sigma, thresh, radius, disp)
%
% Arguments:   
%            im     - image to be processed.
%            sigma  - standard deviation of smoothing Gaussian. Typical
%                     values to use might be 1-3.
%            thresh - threshold (optional). Try a value ~1000.
%            radius - radius of region considered in non-maximal
%                     suppression (optional). Typical values to use might
%                     be 1-3.
%            disp   - optional flag (0 or 1) indicating whether you want
%                     to display corners overlayed on the original
%                     image. This can be useful for parameter tuning. This
%                     defaults to 0
%
% Returns:
%            cim    - binary image marking corners.
%            r      - row coordinates of corner points.
%            c      - column coordinates of corner points.
%            rsubp  - If five return values are requested sub-pixel
%            csubp  - localization of feature points is attempted and
%                     returned as an additional set of floating point
%                     coords. Note that you may still want to use the integer
%                     valued coords to specify centres of correlation windows
%                     for feature matching.
%
% If thresh and radius are omitted from the argument list only 'cim' is returned
% as a raw corner strength image.  You may then want to look at the values
% within 'cim' to determine the appropriate threshold value to use. Note that
% the Harris corner strength varies with the intensity gradient raised to the
% 4th power.  Small changes in input image contrast result in huge changes in
% the appropriate threshold.
%
% Note that this code computes Noble's version of the detector which does not
% require the parameter 'k'.  See comments in code if you wish to use Harris'
% original measure.
%
% See also: NONMAXSUPPTS, DERIVATIVE5

% References: 
% C.G. Harris and M.J. Stephens. "A combined corner and edge detector", 
% Proceedings Fourth Alvey Vision Conference, Manchester.
% pp 147-151, 1988.
%
% Alison Noble, "Descriptions of Image Surfaces", PhD thesis, Department
% of Engineering Science, Oxford University 1989, p45.

% Copyright (c) 2002-2010 Peter Kovesi
% Centre for Exploration Targeting
% The University of Western Australia
% http://www.csse.uwa.edu.au/~pk/research/matlabfns/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% March    2002 - Original version
% December 2002 - Updated comments
% August   2005 - Changed so that code calls nonmaxsuppts
% August   2010 - Changed to use Farid and Simoncelli's derivative filters

function [cim, r, c, rsubp, csubp] = harris(im, sigma, thresh, radius, disp)
    
    error(nargchk(2,5,nargin));
    if nargin == 4
	disp = 0;
    end
    
    if ~isa(im,'double')
	im = double(im);
    end

    subpixel = nargout == 5;

    % Compute derivatives and elements of the structure tensor.
    [Ix, Iy] = derivative5(im, 'x', 'y');
    Ix2 = gaussfilt(Ix.^2,  sigma);
    Iy2 = gaussfilt(Iy.^2,  sigma);    
    Ixy = gaussfilt(Ix.*Iy, sigma);    

    % Compute the Harris corner measure. Note that there are two measures
    % that can be calculated.  I prefer the first one below as given by
    % Nobel in her thesis (reference above).  The second one (commented out)
    % requires setting a parameter, it is commonly suggested that k=0.04 - I
    % find this a bit arbitrary and unsatisfactory. 

    cim = (Ix2.*Iy2 - Ixy.^2)./(Ix2 + Iy2 + eps); % My preferred  measure.
%    k = 0.04;
%    cim = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2; % Original Harris measure.

    if nargin > 2   % We should perform nonmaximal suppression and threshold

	if disp  % Call nonmaxsuppts to so that image is displayed
	    if subpixel
		[r,c,rsubp,csubp] = nonmaxsuppts(cim, radius, thresh, im);
	    else
		[r,c] = nonmaxsuppts(cim, radius, thresh, im);		
	    end
	else     % Just do the nonmaximal suppression
	    if subpixel
		[r,c,rsubp,csubp] = nonmaxsuppts(cim, radius, thresh);
	    else
		[r,c] = nonmaxsuppts(cim, radius, thresh);		
	    end
	end
    end
    
