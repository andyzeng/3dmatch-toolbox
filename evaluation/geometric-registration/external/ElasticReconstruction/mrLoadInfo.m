function [ info ] = mrLoadInfo( filename )
    fid = fopen( filename );
    k = 1;
    x = fscanf( fid, '%d', [ 1, 3 ] );
    while ( size( x, 2 ) == 3 )
        m = fscanf( fid, '%f', [ 6, 6 ] );
        info( k ) = struct( 'info', x, 'mat', m' );
        k = k + 1;
        x = fscanf( fid, '%d', [ 1, 3 ] );
    end
    fclose( fid );
    %disp( [ num2str( size( info, 2 ) ), ' matrices have been read.' ] );
end
