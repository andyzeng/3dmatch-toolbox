% DERIVATIVE5 - 5-Tap 1st and 2nd discrete derivatives
%
% This function computes 1st and 2nd derivatives of an image using the 5-tap
% coefficients given by Farid and Simoncelli.  The results are significantly
% more accurate than MATLAB's GRADIENT function on edges that are at angles
% other than vertical or horizontal. This in turn improves gradient orientation
% estimation enormously.  If you are after extreme accuracy try using DERIVATIVE7.
%
% Usage:  [gx, gy, gxx, gyy, gxy] = derivative5(im, derivative specifiers)
%
% Arguments:
%                       im - Image to compute derivatives from.
%    derivative specifiers - A comma separated list of character strings
%                            that can be any of 'x', 'y', 'xx', 'yy' or 'xy'
%                            These can be in any order, the order of the
%                            computed output arguments will match the order
%                            of the derivative specifier strings.
% Returns:
%  Function returns requested derivatives which can be:
%     gx, gy   - 1st derivative in x and y
%     gxx, gyy - 2nd derivative in x and y
%     gxy      - 1st derivative in y of 1st derivative in x
%
%  Examples:
%    Just compute 1st derivatives in x and y
%    [gx, gy] = derivative5(im, 'x', 'y');  
%                                           
%    Compute 2nd derivative in x, 1st derivative in y and 2nd derivative in y
%    [gxx, gy, gyy] = derivative5(im, 'xx', 'y', 'yy')
%
% See also: DERIVATIVE7

% Reference: Hany Farid and Eero Simoncelli "Differentiation of Discrete
% Multi-Dimensional Signals" IEEE Trans. Image Processing. 13(4): 496-508 (2004)

% Copyright (c) 2010 Peter Kovesi
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
%
% April 2010

function varargout = derivative5(im, varargin)

    varargin = varargin(:);
    varargout = cell(size(varargin));
    
    % Check if we are just computing 1st derivatives.  If so use the
    % interpolant and derivative filters optimized for 1st derivatives, else
    % use 2nd derivative filters and interpolant coefficients.
    % Detection is done by seeing if any of the derivative specifier
    % arguments is longer than 1 char, this implies 2nd derivative needed.
    secondDeriv = false;    
    for n = 1:length(varargin)
        if length(varargin{n}) > 1
            secondDeriv = true;
            break
        end
    end
    
    if ~secondDeriv
        % 5 tap 1st derivative cofficients.  These are optimal if you are just
        % seeking the 1st deriavtives
        p = [0.037659  0.249153  0.426375  0.249153  0.037659];
        d1 =[0.109604  0.276691  0.000000 -0.276691 -0.109604];
    else         
        % 5-tap 2nd derivative coefficients. The associated 1st derivative
        % coefficients are not quite as optimal as the ones above but are
        % consistent with the 2nd derivative interpolator p and thus are
        % appropriate to use if you are after both 1st and 2nd derivatives.
        p  = [0.030320  0.249724  0.439911  0.249724  0.030320];
        d1 = [0.104550  0.292315  0.000000 -0.292315 -0.104550];
        d2 = [0.232905  0.002668 -0.471147  0.002668  0.232905];
    end

    % Compute derivatives.  Note that in the 1st call below MATLAB's conv2
    % function performs a 1D convolution down the columns using p then a 1D
    % convolution along the rows using d1. etc etc.
    gx = false;
    
    for n = 1:length(varargin)
      if strcmpi('x', varargin{n})
          varargout{n} = conv2(p, d1, im, 'same');    
          gx = true;   % Record that gx is available for gxy if needed
          gxn = n;
      elseif strcmpi('y', varargin{n})
          varargout{n} = conv2(d1, p, im, 'same');
      elseif strcmpi('xx', varargin{n})
          varargout{n} = conv2(p, d2, im, 'same');    
      elseif strcmpi('yy', varargin{n})
          varargout{n} = conv2(d2, p, im, 'same');
      elseif strcmpi('xy', varargin{n}) | strcmpi('yx', varargin{n})
          if gx
              varargout{n} = conv2(d1, p, varargout{gxn}, 'same');
          else
              gx = conv2(p, d1, im, 'same');    
              varargout{n} = conv2(d1, p, gx, 'same');
          end
      else
          error(sprintf('''%s'' is an unrecognized derivative option',varargin{n}));
      end
    end
    
