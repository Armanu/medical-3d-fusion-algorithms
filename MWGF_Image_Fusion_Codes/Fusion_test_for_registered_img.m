% script_fusionOthers.m
% -------------------------------------------------------------------
% 
% Date:    10/04/2013
% Last modified: 1/11/2013
% -------------------------------------------------------------------

% function Fusion_test_for_registered_img()

%     clear
%     close all
%     clc

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
% for i=1:10
    %% ------ Input the images ----------------
    % ------------- The Gray ----------------

%     path1 = ['./MF_images/image',num2str(i),'_left.png'];
%     path2 = ['./MF_images/image',num2str(i),'_right.png'];
%     fused_path = ['./fused_mf/fused',num2str(i),'_mwgf.png'];

%     path1 = ['./mf_noise_images/image',num2str(i),label,'_left.png'];
%     path2 = ['./mf_noise_images/image',num2str(i),label,'_right.png'];
%     fused_path = ['./fused_mf_noise/fused',num2str(i),label,'_mwgf.png'];

V = niftiread('./HANCT.nii');
[ri,ci,si] = size(V);
disp(si)
disp(ci)
disp(ri)
P = niftiread('./HANPT.nii');
T=P;
for s = 1:si
V1 = double(squeeze(V(:,:,s)));
%V1 = cat(3, V1, V1, V1);
V2 = V1;
for r = 1:ri
    for c = 1:ci
        res = round(abs(((V1(r,c)+1000)/2000)*255));
        if (res > 255)
            V2(r,c) = 255;
        elseif (res < 0)
            V2(r,c) = 0;
        else
            V2(r,c) = res;
        end
    end
end


P1 = double(squeeze(P(:,:,s)));
%P1 = cat(3, P1, P1, P1);
P2 = P1;
for pr = 1:ri
    for pc = 1:ci
        pres = round(abs((P1(pr,pc)/4)*255));
        if (pres > 255)
            P2(pr,pc) = 255;
        elseif (pres < 0)
            P2(pr,pc) = 0;
        else
            P2(pr,pc) = pres;
        end
        %disp(P2(pr,pc));
    end
end
    path1 = ['./registered-images/image7_left_re.png'];
    path2 = ['./registered-images/image7_right_re.png'];
    fused_path = ['./fused_re/fused7_re_mwgf.png'];

    % -----------------------------------------
    [img1, img2] = PickName(path1, path2, 0);
    paraShow.fig = 'Input 1';
    paraShow.title = 'Org1';
    %ShowImageGrad(V1, paraShow)
    paraShow.fig = 'Input 2';
    paraShow.title = 'Org2';
    %ShowImageGrad(P1, paraShow)
    %% ---- The parameters -----
    % ----------- the multi scale -----
    para.Scale.lsigma = 4;
    para.Scale.ssigma = 0.5;
    para.Scale.alpha = 0.5;
    % -------------- the Merge parameter fusion of registered images-------------
    para.Merge.per = 0.5;
    para.Merge.margin = 1.5*para.Scale.lsigma;
    para.Merge.method = 3;
    % ------------- the Reconstruct parameter -----------
    para.Rec.iter = 500;
    para.Rec.res = 1e-6;
    para.Rec.modify = 5;
    para.Rec.iniMode = 'weight';   
    
    %% ---- MWGF implementation ------
    imgRec = MWGFusion(V2, P2, para);

    % --- Show the result ------
%     paraShow.fig = 'fusion result';
%     paraShow.title = 'MWGF';
%     ShowImageGrad(imgRec, paraShow);
%     imwrite(uint8(imgRec), 'result.jpg', 'jpeg');
    imgRecUint = uint8(imgRec);
%     figure;imshow(imgRecUint);
    T(:,:,s) = imgRec;
end
niftiwrite(T,'outbrain.nii');
% end
% end
% end