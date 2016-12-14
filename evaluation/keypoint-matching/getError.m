% Example evaluation script of keypoint matching benchmark to compute
% false-positive rate (error) at 95% recall 

% Path to .log files (change me)
descLogFile = '3dmatch.log'; % List of descriptor distances per comparison
gtLogFile = 'gt.log';

% Load descriptor distances 
descDistPred = dlmread(descLogFile);
numComparisons = descDistPred(1);
descDistPred = descDistPred(2:end)';

% Load ground truth binary labels (1 - is match, 0 - is nonmatch)
descIsMatchGT = dlmread(gtLogFile);
numComparisons = descIsMatchGT(1);
descIsMatchGT = descIsMatchGT(2:end)';

% Loop through 1001 descriptor distance thresholds
thresh = (0:(max(descDistPred)*1.05/1000):(max(descDistPred)*1.05))';

allThresh = repmat(thresh,1,size(descDistPred,2));
allDescDistPred = repmat(descDistPred,length(thresh),1);
allDescIsMatchGT = repmat(descIsMatchGT,length(thresh),1);
allDescIsMatchPred = allDescDistPred < allThresh;

numTP = sum(allDescIsMatchPred&allDescIsMatchGT,2);
numFP = sum(allDescIsMatchPred&~allDescIsMatchGT,2);
numTN = sum(~allDescIsMatchPred&~allDescIsMatchGT,2);
numFN = sum(~allDescIsMatchPred&allDescIsMatchGT,2);

accuracy = (numTP+numTN)./(numTP+numTN+numFP+numFN);
precision = numTP./(numTP+numFP);
recall = numTP./(numTP+numFN);
TNrate = numTN./(numTN+numFP);
FPrate = numFP./(numFP+numTN);

errorAt95Recall = mean(FPrate(find(recall>0.949&recall<0.951)));
fprintf('False-positive rate (error) at 95%% recall: %f\n',errorAt95Recall);

