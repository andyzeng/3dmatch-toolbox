% HCROSS - Homogeneous cross product, result normalised to s = 1.
%
% Function to form cross product between two points, or lines,
% in homogeneous coodinates.  The result is normalised to lie
% in the scale = 1 plane.
% 
% Usage: c = hcross(a,b)
%

% Copyright (c) 2000-2005 Peter Kovesi
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

%  April 2000

function c = hcross(a,b)
c = cross(a,b);
c = c/c(3);
