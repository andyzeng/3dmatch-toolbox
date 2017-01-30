function mrWriteInfo( info, filename )
    fid = fopen( filename, 'w' );
    for i = 1 : size( info, 2 )
        mrWriteInfoStruct( fid, info( i ).info, info( i ).mat );
    end
    fclose( fid );
    %disp( [ num2str( size( info, 2 ) ), ' matrices have been written.' ] );
end

function mrWriteInfoStruct( fid, x, m )
    fprintf( fid, '%d\t%d\t%d\n', x(1), x(2), x(3) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n', ...
        m(1,1), m(1,2), m(1,3), m(1,4), m(1,5), m(1,6) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n', ...
        m(2,1), m(2,2), m(2,3), m(2,4), m(2,5), m(2,6) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n', ...
        m(3,1), m(3,2), m(3,3), m(3,4), m(3,5), m(3,6) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n', ...
        m(4,1), m(4,2), m(4,3), m(4,4), m(4,5), m(4,6) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n', ...
        m(5,1), m(5,2), m(5,3), m(5,4), m(5,5), m(5,6) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\t%.10f\t%.10f\n', ...
        m(6,1), m(6,2), m(6,3), m(6,4), m(6,5), m(6,6) );
end
