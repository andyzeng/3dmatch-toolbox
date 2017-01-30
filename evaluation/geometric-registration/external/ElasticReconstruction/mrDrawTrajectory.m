function mrDrawTraj( traj, c, init_trans )
    if ~exist( 'c', 'var' )
        c = 'b-';
    end

    if ~exist( 'init_trans', 'var' )
        init_trans = traj( 1 ).trans;
    end

    n = size( traj, 2 );
    x = zeros( 2, n );
    init_inverse = init_trans ^ -1;

    for k = 1 : n
        m = init_inverse * traj( k ).trans;
        x( :, k ) = [ m( 1, 4 ); m( 3, 4 ) ];
    end

    plot( -x( 1, : ), -x( 2, : ), c, 'LineWidth',2 );
    axis equal;
end