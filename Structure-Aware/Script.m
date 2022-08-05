clear;
close all
addpath(genpath(pwd));
%%----------------START OF NIFTY FOLDER LOOP-----------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%

ctfilelist = dir(fullfile('./CT_images/', '*.nii'));
petfilelist = dir(fullfile('./PET_images/', '*.nii'));

for ctfile = ctfilelist
V = niftiread('./CT_images/'+string(ctfile.name));
P = niftiread('./PET_images/'+string(ctfile.name));
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

%     J(:,:,1) = V1;
%     J(:,:,2) = P1;
    F_echo_dtf = IJF(V1,P1);
    %imwrite(F_echo_dtf,[num2str(i),'.bmp']);
    
T(:,:,s) = F_echo_dtf;
clear J;
end
niftiwrite(T,'OUTPUT/'+string(ctfile.name));
end
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------END OF NIFTY FOLDER LOOP------------------%%

%Q_echo_dtf = Qp_ABF(V1, P1, F_echo_dtf)
