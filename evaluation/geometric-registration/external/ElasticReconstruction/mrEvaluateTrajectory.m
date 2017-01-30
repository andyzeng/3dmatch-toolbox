function [ rmse, trans ] = mrEvaluateTraj( traj_et, traj_gt )
    gt_n = size( traj_gt, 2 );
    et_n = size( traj_et, 2 );
    if (gt_n ~= et_n)
        fprintf('WARNING: There are Lost Frames!\n');
        fprintf('ground truth traj : %d frames\n', gt_n);
        fprintf('estimated traj    : %d frames\n', et_n);
        gt_n = min( [ gt_n, et_n ] );
        et_n = gt_n;
    end
    n = et_n;

    trans = mrAlignTraj( traj_et, traj_gt );
    err = zeros( 1, n );
    
    for i = 1 : n
        assert( traj_et( i ).info( 3 ) == traj_gt( i ).info( 3 ),...
            'bad trajectory file format or asynchronized frame.' );
        trans_et = trans * traj_et( i ).trans;
        trans_gt = traj_gt( i ).trans;
        err( i ) = norm( trans_gt( 1 : 3, 4 ) - trans_et( 1 : 3, 4 ) );
    end
    
    rmse = sqrt( err * err' / size( err, 2 ) );
    fprintf( 'median absolute translational error %f m\n', median( err ) );
    fprintf( 'rmse %f m\n', rmse );
end

function [ trans ] = mrAlignTraj( traj_et, traj_gt )
    n = size( traj_et, 2 );
    gt_trans = zeros( 3, n );
    et_trans = zeros( 3, n );

    for i = 1 : n
        gt_trans( :, i ) = traj_gt( i ).trans( 1 : 3, 4 );
        et_trans( :, i ) = traj_et( i ).trans( 1 : 3, 4 );
    end

    gt_mean = mean( gt_trans, 2 );
    et_mean = mean( et_trans, 2 );
    gt_centered = gt_trans - repmat( gt_mean, 1, n );
    et_centered = et_trans - repmat( et_mean, 1, n );

    W = zeros( 3, 3 );
    for i = 1 : n
        W = W + et_centered( :, i ) * gt_centered( :, i )';
    end

    [ U, ~, V ] = svd( W' );
    Vh = V';
    S = eye( 3 );
    if ( det( U ) * det( Vh ) < 0 )
        S( 3, 3 ) = -1;
    end

    r = U * S * Vh;
    t = gt_mean - r * et_mean;

    trans = [ r, t; 0, 0, 0, 1 ];
end