
% ========================================================================
% Fast Multi-Scale Structural Patch Decomposition for Multi-Exposure Image Fusion, TIP,2020
% algorithm Version 1.0
% Copyright(c) 2020, Hui Li, Kede Ma, Yongwei Yong and Lei Zhang
% All Rights Reserved.
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
% This is a static scene implementation of "Fast Multi-Scale Structural Patch Decomposition for Multi-Exposure Image Fusion"
% Please refer to the following paper:
% H. Li et al., "Fast Multi-Scale Structural Patch Decomposition for Multi-Exposure Image Fusion, 2020" In press
% IEEE Transactions on Image Processing
% Please kindly report any suggestions or corrections to xiaohui102788@126.com
%----------------------------------------------------------------------
clear;
close all;
addpath(genpath(pwd));
V = niftiread('./HANCT.nii');
P = niftiread('./HANPT.nii');
PF = P(:);

[x, y, z] = ind2sub(size(P), find(P == max(P(:))));
val = P(x,y,z);
PMA = max(PF);
PMI = min(PF);
disp(PMI)
disp(PMA)
disp("-------")

VF = V(:);
VMA = max(VF);
VMI = min(VF);
disp(VMI)
disp(VMA)
disp("-------")
[ri,ci,si] = size(V);
for s = 1:si
V1 = (double(squeeze(V(:,:,s)))/(VMA-VMI))*255;
P1 = (double(squeeze(P(:,:,s)))/(PMA-PMI))*255;
V13 = cat(3, V1, V1, V1);
P13 = cat(3, P1, P1, P1);
imwrite(V13,'New Folder/c.png','png');
imwrite(P13,'New Folder/p.png','png');

%static scenes
imgSeqColor= loadImg('New Folder',1); % [0,1]
%disp(imgSeqColor.shape)
%     imgSeqColor = downSample(imgSeqColor, 1024);

%% the finest scale
tic
r1=4;
[ D1,i_mean1,aa1,N1] = scale_fine(imgSeqColor,r1);

%% the intermediate  scale
[w,h,~,~]=size(imgSeqColor);
nlev = floor(log(min(w,h)) / log(2))-5;

D2 = cell(nlev,1);
aa2= cell(nlev,1);
N2= cell(nlev,1);

r2=4;
for ii=1:nlev
%     disp(ii)
    [ D2{ii},i_mean2,aa2{ii},N2{ii}] = scale_interm(i_mean1,r2);
    i_mean1=i_mean2;
end


%% the coarsest  scale
r3=4;
[fI3,i_mean3,aa3,N3] = scale_coarse(i_mean2,r3);

%% reconstruct
%% Intermediate layers
for ii=nlev:-1:1
    temp=aa2{ii};
    fI=zeros(size(temp));
    fI(1:2:size(temp,1),1:2:size(temp,2))=fI3;
    B2=boxfilter(fI, r2)./ N2{ii}+D2{ii};
    
    fI3=B2;
end
%% finest layers
fI=zeros(size(aa1));
fI(1:2:size(aa1,1),1:2:size(aa1,2))=B2;
B1=boxfilter(fI, r1)./ N1;
C_out=B1+rgb2gray(D1);
toc

%figure,imshow(C_out)
delete 'New Folder/p.png'
delete 'New Folder/c.png'
T(:,:,s) = C_out;

end
niftiwrite(T,'outbrain.nii');
T2 = T(:);