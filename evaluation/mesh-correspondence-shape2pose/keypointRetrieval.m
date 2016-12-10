% Visualize keypoint correspondences between two object models
%
% ---------------------------------------------------------
% Copyright (c) 2016, Andy Zeng
% 
% This file is part of the 3DMatch Toolbox and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% Path to Shape2Pose data (change me)
dataPath1 = '../../data/shape2pose/bikes';
dataPath2 = '../../data/shape2pose/bikes';

objectFiles1 = dir(fullfile(dataPath1,'*.raw'));
objectFiles2 = dir(fullfile(dataPath2,'*.raw'));
for objectIdx = [1,10]

    object1Prefix = objectFiles1(1).name(1:(end-4));
    object2Prefix = objectFiles2(objectIdx).name(1:(end-4));
    
    fprintf('%s %s\n',object1Prefix,object2Prefix);

%     % Bicycle
%     interestPoint2 = [169,99,176]; % left handle
%     interestPoint = [140,55,92]; % right footstep
    interestPoint1 = [123,68,137]; % chair
    interestPoint2 = [255,72,78]; % front fender

%     % Chair
%     interestPoint1 = [103,85,98]; % seat diagonal
%     interestPoint2 = [43,95,43]; % leg foot

%     % Chair wheels
%     interestPoint1 = [41,67,45];
%     interestPoint2 = [72,76,67];
    
%     % Cart
%     interestPoint1 = [72,83,89]; % wheel center 4
%     interestPoint1 = [101,425,126]; % handle end 4
%     interestPoint2 = [101,425,126]; % handle end 4
%     interestPoint1 = [81,49,45]; % front cart wheel middle 6
%     interestPoint2 = [81,49,45]; % front cart wheel middle 6
%     interestPoint1 = [145,57,128]; % front cart corner 6
%     interestPoint2 = [145,57,128]; % front cart corner 6
%     interestPoint1 = [85,162,132]; % side 4
%     interestPoint2 = [85,162,132]; % side 4
%     interestPoint1 = [67,88,40]; % front cart wheel bottom 4
%     interestPoint2 = [67,88,40]; % front cart wheel bottom 4

    % Load keypoints from object 1
    fid = fopen(fullfile(dataPath1,sprintf('%s.keypoints.bin',object1Prefix)),'rb');
    object1NumKeypoints = fread(fid,1,'single');
    object1Keypoints = fread(fid,'single');
    object1Keypoints = reshape(object1Keypoints,3,object1NumKeypoints)';
    fclose(fid);

    % Load 3DMatch feature descriptors for keypoints from object 1
    fid = fopen(fullfile(dataPath1,sprintf('%s.keypoints.3dmatch.descriptors.bin',object1Prefix)),'rb');
    object1DescriptorData = fread(fid,'single');
    object1NumDescriptors = object1DescriptorData(1);
    object1DescriptorSize = object1DescriptorData(2);
    object1Descriptors = reshape(object1DescriptorData(3:end),object1DescriptorSize,object1NumDescriptors)';
    fclose(fid);

    % Load keypoints from object 2
    fid = fopen(fullfile(dataPath2,sprintf('%s.keypoints.bin',object2Prefix)),'rb');
    object2NumKeypoints = fread(fid,1,'single');
    object2Keypoints = fread(fid,'single');
    object2Keypoints = reshape(object2Keypoints,3,object2NumKeypoints)';
    fclose(fid);

    % Load 3DMatch feature descriptors for keypoints from object 2
    fid = fopen(fullfile(dataPath2,sprintf('%s.keypoints.3dmatch.descriptors.bin',object2Prefix)),'rb');
    object2DescriptorData = fread(fid,'single');
    object2NumDescriptors = object2DescriptorData(1);
    object2DescriptorSize = object2DescriptorData(2);
    object2Descriptors = reshape(object2DescriptorData(3:end),object2DescriptorSize,object2NumDescriptors)';
    fclose(fid);

    % Compute descriptor distances
    interestPoint1Idx = find(sum(abs(object1Keypoints-repmat(interestPoint1,object1NumKeypoints,1)),2) == 0);
    interestPoint1Descriptor = object1Descriptors(interestPoint1Idx,:);
    distances1 = sqrt(sum((repmat(interestPoint1Descriptor,object2NumDescriptors,1)-object2Descriptors).^2,2));
    interestPoint2Idx = find(sum(abs(object1Keypoints-repmat(interestPoint2,object1NumKeypoints,1)),2) == 0);
    interestPoint2Descriptor = object1Descriptors(interestPoint2Idx,:);
    distances2 = sqrt(sum((repmat(interestPoint2Descriptor,object2NumDescriptors,1)-object2Descriptors).^2,2));
    
    truncation1 = 0.25;
    truncation2 = 0.25;
    distances1(find(distances1 > truncation1)) = truncation1;
    distances1 = 1 - distances1./truncation1;
    distances2(find(distances2 > truncation2)) = truncation2;
    distances2 = 1 - distances2./truncation2;
    [maxDistance,maxIdx] = max([distances1,distances2],[],2);
    
    % Visualize distances on object models
    colors = ones(size(distances1,1),3)*180;
    validRed = find(maxIdx == 1 & distances1 > 0);
    validBlue = find(maxIdx == 2 & distances2 > 0);
    colors(validRed,:) = colors(validRed,:)+round([distances1(validRed)./max(distances1(validRed))*75, ...
                                                  -distances1(validRed)./max(distances1(validRed))*180, ...
                                                  -distances1(validRed)./max(distances1(validRed))*180]);
    colors(validBlue,:) = colors(validBlue,:)+round([-distances2(validBlue)./max(distances2(validBlue))*180, ...
                                                     -distances2(validBlue)./max(distances2(validBlue))*180, ...
                                                      distances2(validBlue)./max(distances2(validBlue))*75]);
%     pcwrite(pointCloud(object2Keypoints,'Color',uint8(colors)),sprintf('compare%02d',objectIdx),'PLYformat','binary');
    figure(); pcshow(pointCloud(object2Keypoints,'Color',uint8(colors)));
end

