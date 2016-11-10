fp = fopen('keypoint-FPFH.dat','rb');
data = fread(fp,'single');
fclose(fp);

numDescriptors = data(1);
descriptorSize = data(2);
descriptors = reshape(data(3:end),descriptorSize,numDescriptors)';
% descriptors = reshape(data(3:end),numDescriptors,descriptorSize); <- transpose

posCorresDistances = sqrt(sum((descriptors(1:3:numDescriptors,:)-descriptors(2:3:numDescriptors,:)).^2,2));
negCorresDistances = sqrt(sum((descriptors(1:3:numDescriptors,:)-descriptors(3:3:numDescriptors,:)).^2,2));
distancePredictions = [posCorresDistances',negCorresDistances'];

isMatchGroundTruth = [ones(size(posCorresDistances))',zeros(size(negCorresDistances))'];
thresholds = (0:0.1:210)';

repeatThresholds = repmat(thresholds,1,size(distancePredictions,2));
repeatDistancePredictions = repmat(distancePredictions,length(thresholds),1);
repeatIsMatchGroundTruth = repmat(isMatchGroundTruth,length(thresholds),1);

repeatIsMatchPredictions = repeatDistancePredictions < repeatThresholds;

numTP = sum(repeatIsMatchPredictions&repeatIsMatchGroundTruth,2);
numFP = sum(repeatIsMatchPredictions&~repeatIsMatchGroundTruth,2);
numTN = sum(~repeatIsMatchPredictions&~repeatIsMatchGroundTruth,2);
numFN = sum(~repeatIsMatchPredictions&repeatIsMatchGroundTruth,2);

accuracy = (numTP+numTN)./(numTP+numTN+numFP+numFN);
precision = numTP./(numTP+numFP);
recall = numTP./(numTP+numFN);
TNrate = numTN./(numTN+numFP);
FPrate = numFP./(numFP+numTN);

precisionAt95Recall = mean(precision(find(recall>0.949&recall<0.951)))

