
% Load dictionary
% load('dict.mat');  
% n = 20;
% time = zeros(n,1);
% 
% label='';
% for k = 1:5
% if k==1
%     label='_gau_001';
% end
% if k==2
%     label='_gau_0005';
% end
% if k==3
%     label='_sp_01';
% end
% if k==4
%     label='_sp_02';
% end
% if k==5
%     label='_poi';
% end
% disp(label);

V = niftiread('./HANCT.nii');
P = niftiread('./HANPT.nii');
PF = P(:);
PMA = max(PF);
PMI = min(PF);
VF = V(:);
VMA = max(VF);
VMI = min(VF);
[ri,ci,si] = size(V);
for s = 1:si
V1 = (double(squeeze(V(:,:,s)))/(VMA-VMI))*255;
P1 = (double(squeeze(P(:,:,s)))/(PMA-PMI))*255;

% path1 = ['./IV_images/IR',num2str(index),'.png'];
% path2 = ['./IR_images/VIS',num2str(index),'.png'];
% fused_path = ['./fused_infrared/fused',num2str(index),'_ConvSR.png'];

% path1 = ['./MF_images/image',num2str(index),'_left.png'];
% path2 = ['./MF_images/image',num2str(index),'_right.png'];
% fused_path = ['./fused_mf/fused',num2str(index),'_ConvSR.png'];
% 
% path1 = ['../../_________________________DATA/mid/Test_ir/',num2str(i),'.bmp'];
% path2 = ['../../_________________________DATA/mid/Test_ir/',num2str(i),'.bmp'];
% fused_path = ['../..//3/',num2str(i),label,'_convsr.bmp'];

% Load images
% A=imread('sourceimages/s01_1.tif');
% B=imread('sourceimages/s01_2.tif');
A=V1;
B=P1;

% figure,imshow(A)
% figure,imshow(B)

%key parameters
lambda=0.01; 
flag=1; % 1 for multi-focus image fusion and otherwise for multi-modal image fusion

%CSR-based fusion
tic;
F=CSR_Fusion(A,B,3,lambda,flag);
time(i) = toc;

% % figure,imshow(F);
% imwrite(F,fused_path);
% 
% end
% end
T(:,:,s) = F;
end
niftiwrite(T,'outbrain.nii');