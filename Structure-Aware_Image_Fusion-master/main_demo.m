clear;
close all
addpath(genpath(pwd));
V = niftiread('./HANCT.nii');
[ri,ci,si] = size(V);
P = niftiread('./HANPT.nii');
for s = 1:si
V1 = (double(squeeze(V(:,:,s)))/1000)*255;
P1 = (double(squeeze(P(:,:,s)))/4)*255;
%     J(:,:,1) = V1;
%     J(:,:,2) = P1;
    F_echo_dtf = IJF(V1,P1);
    %imwrite(F_echo_dtf,[num2str(i),'.bmp']);
    
T(:,:,s) = F_echo_dtf;
clear J;
end
niftiwrite(T,'outbrain.nii');
%Q_echo_dtf = Qp_ABF(V1, P1, F_echo_dtf)
