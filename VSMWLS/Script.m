% This code is in association with the following paper
% "Ma J, Zhou Z, Wang B, et al. Infrared and visible image fusion based on visual saliency map and weighted least square optimization[J].
% Infrared Physics & Technology, 2017, 82:8-17."
% Authors: Jinlei Ma, Zhiqiang Zhou, Bo Wang, Hua Zong
% Code edited by Jinlei Ma, email: majinlei121@163.com

clear all
close all

%%----------------START OF NIFTY FOLDER LOOP-----------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%

%%----------------START OF NIFTY FOLDER LOOP-----------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%


ctfilelist = dir(fullfile('G:Croped\CT\*\', '*.nii'));

for i = 1:length(ctfilelist)
    newfile = split(string(ctfilelist(i).name),'.');
    newfile = string(newfile{1});
    %disp(newfile)
    newfolder = 'G:\VSMWLS\'+newfile+'\';
    if ~exist(newfolder, 'dir')
        if exist('G:\Croped\PET\'+newfile+'\'+string(ctfilelist(i).name), 'file')
            V = niftiread('G:\Croped\CT\'+newfile+'\'+string(ctfilelist(i).name));
            P = niftiread('G:\Croped\PET\'+newfile+'\'+string(ctfilelist(i).name));
            clear T;

            if ~exist(newfolder,'dir')
                mkdir(newfolder)
            end

            disp("file : "+ string(ctfilelist(i).name))
            [ri,ci,si] = size(V);
            for s = 1:si
                V1 = uint8(squeeze(V(:,:,s))*255);
                P1 = uint8(squeeze(P(:,:,s))*255);

                % path1 = ['./MF_images/image',num2str(index),'_left.png'];
                % path2 = ['./MF_images/image',num2str(index),'_right.png'];
                % fused_path = ['./fused_mf/fused',num2str(index),'_wls.png'];

                %path1 = ['./mf_noise_images/image',num2str(i),label,'_left.png'];
                %path2 = ['./mf_noise_images/image',num2str(i),label,'_right.png'];
                %fused_path = ['./fused_mf_noise/fused1',num2str(i),label,'_wls.png'];

                % I1 is a visible image, and I2 is an infrared image.
                %I1 = imread(path1);
                %I2 = imread(path2);

                I1 = im2double(V1);
                I2 = im2double(P1);

                % figure;imshow(I1);
                % figure;imshow(I2);
                tic
                fused = WLS_Fusion(I1,I2);
                toc

                % figure;imshow(fused);
                %imwrite(fused,fused_path,'png');
                T(:,:,s) = fused;
            end

            niftiwrite(T,newfolder+newfile+'.nii');

        end
    end
end
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------END OF NIFTY FOLDER LOOP------------------%%

