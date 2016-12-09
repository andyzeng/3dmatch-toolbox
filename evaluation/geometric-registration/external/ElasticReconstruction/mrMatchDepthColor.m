function mrMatchDepthColor( basepath, unique, depthdir, imagedir, matchfile )
    if ~exist( 'matchfile', 'var' )
        matchfile = 'match';
    end
    if ~exist( 'imagedir', 'var' )
        imagedir = 'rgb';
    end
    if ~exist( 'depthdir', 'var' )
        depthdir = 'depth';
    end
    if ~exist( 'unique', 'var' )
        unique = 1;
    end

    depth_file_list = dir( [ basepath, depthdir, '/*.png' ] );
    if ( size( depth_file_list, 1 ) <= 1 )
        disp( 'Error: path not found' );
        return;
    end
    disp( [ num2str( size( depth_file_list, 1 ) ) ' depth images detected.' ] );
    depth_timestamp = parseTimestamp( depth_file_list );

    color_file_list = dir( [ basepath, imagedir, '/*.jpg' ] );
    if ( size( color_file_list, 1 ) <= 1 )
        disp( 'Error: path not found' );
        return;
    end
    disp( [ num2str( size( color_file_list, 1 ) ) ' color images detected.' ] );
    color_timestamp = parseTimestamp( color_file_list );
    color_timestamp_mat = cell2mat( color_timestamp( :, 1 ) );

    fid = fopen( [ basepath, matchfile ], 'w' );
    used_color = zeros( size( color_timestamp, 1 ), 1 );
    k = 0;
    for i = 1 : size( depth_timestamp, 1 )
        idx = findClosestColor( depth_timestamp{ i, 1 }, color_timestamp_mat );
        if ( unique == 0 || used_color( idx ) == 0 )
            used_color( idx ) = 1;
            fprintf( fid, '%s/%s %s/%s\n', depthdir, depth_timestamp{ i, 2 }, imagedir, color_timestamp{ idx, 2 } );
            k = k + 1;
        end
    end
    fclose( fid );
    disp( [ num2str( k ) ' pairs have been written.' ] );
end

function [ i ] = findClosestColor( depth_ts, color_ts_mat )
    [ ~, i ] = min( abs( color_ts_mat - depth_ts ) );
end

function [ timestamp ] = parseTimestamp( filelist )
    num = size( filelist, 1 );
    timestamp = cell( num, 2 );
    for i = 1 : num
        x = sscanf( filelist( i ).name, '%f-%f.' )';
        timestamp{ i, 1 } = x( 2 );
        timestamp{ i, 2 } = filelist( i ).name;
    end
    sortrows( timestamp );
end