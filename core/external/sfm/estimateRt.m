% Usage:   Rt = estimateRt(x1, x2)
%          Rt = estimateRt(x)
%
% Arguments:
%          x1, x2 - Two sets of corresponding 3xN set of homogeneous
%          points.
%         
%          x      - If a single argument is supplied it is assumed that it
%                   is in the form x = [x1; x2]
% Returns:
%          Rt    - The rotation matrix such that x1 = R * x2 + t

function Rt = estimateRt(x, npts)
    [T, Eps] = estimateRigidTransform(x(1:3,:), x(4:6,:));
    Rt = T(1:3,:);
end
    

