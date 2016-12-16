% a short script to desmonstrate estimateRigidTransform
% Babak Taati
% 2010

%% load sample data 
load sampleData; % loads two 3-D points clouds (P1h, and P2h, in homogeneous coordinates) and the ground truth alignment between them (T, 4x4)

alignedP1h = T * P1h;

subplot(1,3,1); 
cameratoolbar
plot3(alignedP1h(1,:), alignedP1h(2,:), alignedP1h(3,:), 'b.');
hold on
plot3(P2h(1,:), P2h(2,:), P2h(3,:), 'ro');
title('original points, ground truth alignment');

%% add a bit of noise to the second point cloud and estimate the alignment

noisyP1h = P1h + randn(size(P1h))/20; % add some zero mean Gaussian noise
noisyP2h = P2h + randn(size(P2h))/20; % add some zero mean Gaussian noise
% since we have lots of point matches, estimating the rigid transformation works even if the data is very noisy. Try replacing /20 with /5 and see!

noisyP1h(:,4) = 1; % force the last homogenous coordiate to one
noisyP2h(:,4) = 1; % force the last homogenous coordiate to one

[estT, Eps] = estimateRigidTransform(noisyP2h(1:3,:), noisyP1h(1:3,:));

alignedP1h = estT * P1h;
subplot(1,3,2); 
plot3(alignedP1h(1,:), alignedP1h(2,:), alignedP1h(3,:), 'b.');
hold on
plot3(P2h(1,:), P2h(2,:), P2h(3,:), 'ro');
title('original points, estimated alignment from noisy points');


alignedNoisyP1h = estT * noisyP1h;
subplot(1,3,3); 
plot3(alignedNoisyP1h(1,:), alignedNoisyP1h(2,:), alignedNoisyP1h(3,:), 'b.');
hold on
plot3(noisyP2h(1,:), noisyP2h(2,:), noisyP2h(3,:), 'ro');
title('noisy points, estimated alignment from noisy points');
