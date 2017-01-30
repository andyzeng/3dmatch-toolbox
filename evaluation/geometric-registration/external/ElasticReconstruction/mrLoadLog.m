function [ traj ] = mrLoadLog( filename )
    fid = fopen( filename );
    k = 1;
    x = fscanf( fid, '%d', [1 3] );
    while ( size( x, 2 ) == 3 )
        m = fscanf( fid, '%f', [4 4] );
        traj( k ) = struct( 'info', x, 'trans', m' );
        k = k + 1;
        x = fscanf( fid, '%d', [1 3] );
    end
    fclose( fid );
    %disp( [ num2str( size( traj, 2 ) ), ' frames have been read.' ] );
end
