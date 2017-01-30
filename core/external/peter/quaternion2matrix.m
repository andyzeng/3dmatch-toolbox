% QUATERNION2MATRIX - Quaternion to a 4x4 homogeneous transformation matrix
%
%  Usage:  T = quaternion2matrix(Q)
%
%  Argument: Q - a quaternion in the form [w xi yj zk]
%  Returns:  T - 4x4 Homogeneous rotation matrix
% 
% See also MATRIX2QUATERNION, NEWQUATERNION, QUATERNIONROTATE

% Copyright (c) 2008 Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% pk at csse uwa edu au
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

function T = quaternion2matrix(Q)
    
    Q = Q/norm(Q); % Ensure Q has unit norm
    
    % Set up convenience variables
    w = Q(1); x = Q(2); y = Q(3); z = Q(4);
    w2 = w^2; x2 = x^2; y2 = y^2; z2 = z^2;
    xy = x*y; xz = x*z; yz = y*z;
    wx = w*x; wy = w*y; wz = w*z;
    
    T = [w2+x2-y2-z2 , 2*(xy - wz) , 2*(wy + xz) ,  0
         2*(wz + xy) , w2-x2+y2-z2 , 2*(yz - wx) ,  0
         2*(xz - wy) , 2*(wx + yz) , w2-x2-y2+z2 ,  0
              0      ,       0     ,       0     ,  1];
    
