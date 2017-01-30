function mrWriteLog( traj, filename )
    fid = fopen( filename, 'w' );
    for i = 1 : size( traj, 2 )
        mrWriteLogStruct( fid, traj( i ).info, traj( i ).trans );
    end
    fclose( fid );
    %disp( [ num2str( size( traj, 2 ) ), ' frames have been written.' ] );
end

function mrWriteLogStruct( fid, x, m )
    fprintf( fid, '%d\t%d\t%d\n', x(1), x(2), x(3) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\n', m(1,1), m(1,2), m(1,3), m(1,4) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\n', m(2,1), m(2,2), m(2,3), m(2,4) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\n', m(3,1), m(3,2), m(3,3), m(3,4) );
    fprintf( fid, '%.10f\t%.10f\t%.10f\t%.10f\n', m(4,1), m(4,2), m(4,3), m(4,4) );
end
