% HLINE - Plot 2D lines defined in homogeneous coordinates.
%
% Function for ploting 2D homogeneous lines defined by 2 points
% or a line defined by a single homogeneous vector
%
% Usage:   hline(p1,p2)   where p1 and p2 are 2D homogeneous points.
%          hline(p1,p2,'colour_name')  'black' 'red' 'white' etc
%          hline(l)       where l is a line in homogeneous coordinates
%          hline(l,'colour_name')
%

%  Peter Kovesi
%  School of Computer Science & Software Engineering
%  The University of Western Australia
%  pk @ csse uwa edu au
%  http://www.csse.uwa.edu.au/~pk
%
%  April 2000

function hline(a,b,c)

col = 'blue';  % default colour

if nargin >= 2 & isa(a,'double')  & isa(b,'double')   % Two points specified

  p1 = a./a(3);        % make sure homogeneous points lie in z=1 plane
  p2 = b./b(3);

  if nargin == 3 & isa(c,'char')  % 2 points and a colour specified
    col = c;
  end

elseif nargin >= 1 & isa(a,'double')       % A single line specified

  a = a./a(3);   % ensure line in z = 1 plane (not needed??)

  if abs(a(1)) > abs(a(2))   % line is more vertical
    ylim = get(get(gcf,'CurrentAxes'),'Ylim');
    p1 = hcross(a, [0 1 0]');
    p2 = hcross(a, [0 -1/ylim(2) 1]');
  else                       % line more horizontal
    xlim = get(get(gcf,'CurrentAxes'),'Xlim');
    p1 = hcross(a, [1 0 0]');
    p2 = hcross(a, [-1/xlim(2) 0 1]');
  end

  if nargin == 2 & isa(b,'char') % 1 line vector and a colour specified
    col = b;
  end

else
  error('Bad arguments passed to hline');
end

line([p1(1) p2(1)], [p1(2) p2(2)], 'color', col);
