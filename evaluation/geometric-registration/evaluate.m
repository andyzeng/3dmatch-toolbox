% Script to evaluate .log files for the geometric registration benchmarks,
% in the same spirit as Choi et al 2015. Please see:
%
% http://redwood-data.org/indoor/regbasic.html
% https://github.com/qianyizh/ElasticReconstruction/tree/master/Matlab_Toolbox

descriptorName = '3dmatch'; % 3dmatch, spin, fpfh

% Locations of evaluation files
dataPath = '../../data/fragments';

% % Synthetic data benchmark
% sceneList = {'iclnuim-livingroom1-evaluation' ...
%              'iclnuim-livingroom2-evaluation' ...
%              'iclnuim-office1-evaluation' ...
%              'iclnuim-office2-evaluation'};
         
% Real data benchmark
sceneList = {'7-scenes-redkitchen-evaluation', ...
             'sun3d-home_at-home_at_scan1_2013_jan_1-evaluation', ...
             'sun3d-home_md-home_md_scan9_2012_sep_30-evaluation', ...
             'sun3d-hotel_uc-scan3-evaluation', ...
             'sun3d-hotel_umd-maryland_hotel1-evaluation', ...
             'sun3d-hotel_umd-maryland_hotel3-evaluation', ...
             'sun3d-mit_76_studyroom-76-1studyroom2-evaluation', ...
             'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation'};
         
% Load Elastic Reconstruction toolbox
addpath(genpath('external'));

% Compute precision and recall
totalRecall = []; totalPrecision = [];
for sceneIdx = 1:length(sceneList)
    scenePath = fullfile(dataPath,sceneList{sceneIdx});
    
    % Compute registration error
    gt = mrLoadLog(fullfile(scenePath,'gt.log'));
    gt_info = mrLoadInfo(fullfile(scenePath,'gt.info'));
    result = mrLoadLog(fullfile(scenePath,sprintf('%s.log',descriptorName)));
    [recall,precision] = mrEvaluateRegistration(result,gt,gt_info);
    totalRecall = [totalRecall;recall];
    totalPrecision = [totalPrecision;precision];
end
fprintf('Mean registration recall: %f precision: %f\n',mean(totalRecall),mean(totalPrecision));
