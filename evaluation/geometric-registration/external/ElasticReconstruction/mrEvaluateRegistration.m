function [ recall, precision ] = mrEvaluateRegistration( result, gt, gt_info, err2 )
    if ~exist( 'err2', 'var' )
        err2 = 0.04;
    end
    num = gt( 1 ).info( 3 );
    
    mask = zeros( 1, num * num );
    gt_num = 0;
    for i = 1 : size( gt, 2 )
        if ( gt( i ).info( 2 ) - gt( i ).info( 1 ) > 1 )
            mask( gt( i ).info( 1 ) + gt( i ).info( 2 ) * num + 1 ) = i;
            gt_num = gt_num + 1;
        end
    end

    rs_num = 0;
    good = 0;
    bad = 0;
    false_pos = 0;
    error_dis = [];
    for i = 1 : size( result, 2 )
        if ( result( i ).info( 2 ) - result( i ).info( 1 ) > 1 )
            rs_num = rs_num + 1;
            idx = mask( result( i ).info( 1 ) + result( i ).info( 2 ) * num + 1 );
            if idx == 0
                false_pos = false_pos + 1;
            else
                p = mrComputeTransformationError( gt( idx ).trans ^ -1 * result( i ).trans, gt_info( idx ).mat );
                error_dis = [ error_dis, p ];
                if ( p <= err2 )
                    good = good + 1;
                else
                    bad = bad + 1;
                end
            end
        end
    end
    
    recall = good / gt_num;
    precision = good / rs_num;
    %disp( [ 'recall : ' num2str( recall ) ' ( ' num2str( good ) ' / ' num2str( gt_num ) ' )' ] );
    %disp( [ 'precision : ' num2str( precision ) ' ( ' num2str( good ) ' / ' num2str( rs_num ) ' )' ] );
end

function [ p ] = mrComputeTransformationError( trans, info )
    te = trans( 1 : 3, 4 );
    qt = dcm2quat( trans( 1 : 3, 1 : 3 ) );
    er = [ te; - qt( 2 : 4 )' ];
    p = er' * info * er / info( 1, 1 );
end

function [qout] = dcm2quat(DCM)
% this is consistent with the matlab function in
% the Aerospace Toolbox
    qout = zeros(1,4);
    qout(1) = 0.5 * sqrt(1 + DCM(1,1) + DCM(2,2) + DCM(3,3));
    qout(2) = - (DCM(3,2) - DCM(2,3)) / ( 4 * qout(1) );
    qout(3) = - (DCM(1,3) - DCM(3,1)) / ( 4 * qout(1) );
    qout(4) = - (DCM(2,1) - DCM(1,2)) / ( 4 * qout(1) );
end