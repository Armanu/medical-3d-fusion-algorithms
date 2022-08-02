% This is the main function of the paper "Infrared and Visual Image Fusion 
% through Infrared Feature Extraction and Visual Information, Infrared Physics & Technology, 2017."
% Implemented by Zhang Yu (uzeful@163.com).


% clear history and memory
clc,clear,close all;

% para settings
QuadNormDim = 512;
QuadMinDim = 32;
GaussScale = 9;
MaxRatio = 0.001;
StdRatio = 0.8;

% image sets
% names = {'Camp', 'Camp1', 'Dune', 'Gun', 'Navi', 'Kayak', 'Octec', 'Road', 'Road2' 'Steamboat', 'T2', 'T3', 'Trees4906', 'Trees4917'};

V = niftiread('./HANCT.nii');
P = niftiread('./HANPT.nii');
[ri,ci,si] = size(V);
for s = 1:si
V1 = uint8((double(squeeze(V(:,:,s)))/1000)*255);
P1 = uint8((double(squeeze(P(:,:,s)))/4)*255);
% read one image set
% a = 10;
% setName = num2str(a);
% imgVis = imread(strcat('vis\', setName, '.bmp'));
% imgIR = imread(strcat('ir\', setName, '.bmp'));

% image fusion
result = BGR_Fuse(V1, P1, QuadNormDim, QuadMinDim, GaussScale, MaxRatio, StdRatio);

% show image
T(:,:,s) = result;
end
niftiwrite(T,'outbrain.nii');