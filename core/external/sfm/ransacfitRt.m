% Usage:   [Rt, inliers] = ransacfitRt(x1, x2, t)
%
% Arguments:
%          x1  - 3xN set of 3D points.
%          x2  - 3xN set of 3D points such that x1<->x2.
%          t   - The distance threshold between data point and the model
%                used to decide whether a point is an inlier or not.
%
% Note that it is assumed that the matching of x1 and x2 are putative and it
% is expected that a percentage of matches will be wrong.
%
% Returns:
%          Rt      - The 3x4 transformation matrix such that x1 = R*x2 + t.
%          inliers - An array of indices of the elements of x1, x2 that were
%                    the inliers for the best model.
%
% See Also: RANSAC
% Author: Jianxiong Xiao

function [Rt, inliers] = ransacfitRt(x, t, feedback)
    s = 3;  % Number of points needed to fit a Rt matrix.
    
    if size(x,2)==s
        inliers = 1:s;
        Rt = estimateRt(x);
        return;
    end

    fittingfn = @estimateRt;
    distfn    = @euc3Ddist;
    degenfn   = @isdegenerate;
    % x1 and x2 are 'stacked' to create a 6xN array for ransac
    [Rt, inliers] = ransac(x, fittingfn, distfn, degenfn, s, t, feedback);

    % Now do a final least squares fit on the data points considered to
    % be inliers.
    Rt = estimateRt(x(:,inliers));
end

%--------------------------------------------------------------------------
% Note that this code allows for Rt being a cell array of matrices of
% which we have to pick the best one.

function [bestInliers, bestRt] = euc3Ddist(Rt, x, t)
    if iscell(Rt)  % We have several solutions each of which must be tested
        nRt = length(Rt);   % Number of solutions to test
        bestRt = Rt{1};     % Initial allocation of best solution
        ninliers = 0;     % Number of inliers
        for k = 1:nRt
            d =  sum((x(1:3,:) - (Rt{k}(:,1:3)*x(4:6,:)+repmat(Rt{k}(:,4),1,size(x,2)))).^2,1).^0.5;
            inliers = find(abs(d) < t);     % Indices of inlying points
            if length(inliers) > ninliers   % Record best solution
                ninliers = length(inliers);
                bestRt = Rt{k};
                bestInliers = inliers;
            end
        end

    else     % We just have one solution
        d =  sum((x(1:3,:) - (Rt(:,1:3)*x(4:6,:)+repmat(Rt(:,4),1,size(x,2)))).^2,1).^0.5;
        bestInliers = find(abs(d) < t);     % Indices of inlying points
        bestRt = Rt;                        % Copy Rt directly to bestRt
    end
end

%----------------------------------------------------------------------
% (Degenerate!) function to determine if a set of matched points will result
% in a degeneracy in the calculation of a fundamental matrix as needed by
% RANSAC.  This function assumes this cannot happen...

function r = isdegenerate(x)
    r = 0;
end
