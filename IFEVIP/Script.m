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

%%----------------START OF NIFTY FOLDER LOOP-----------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%


ctfilelist = dir(fullfile('C:\Users\Administrator\Downloads\CT\CT\', '*.nii.gz'));

for i = 1:length(ctfilelist)
    newfile = split(string(ctfilelist(i).name),'.');
    newfile = string(newfile{1});
    newfolder = 'C:\Users\Administrator\Documents\GitHub\medical-3d-fusion-algorithms\IFEVIP\OUTPUT\'+newfile+'\';
    if ~exist(newfolder, 'dir')
        if exist('C:\Users\Administrator\Downloads\PET\PET\'+string(ctfilelist(i).name), 'file')
            V = niftiread('C:\Users\Administrator\Downloads\CT\CT\'+string(ctfilelist(i).name));
            P = niftiread('C:\Users\Administrator\Downloads\PET\PET\'+string(ctfilelist(i).name));
            clear T;

            if ~exist(newfolder,'dir')
                mkdir(newfolder)
            end

            disp("file : "+ string(ctfile(i).name))


            PF = P(:);
            PMA = max(PF);
            PMI = min(PF);
            VF = V(:);
            VMA = max(VF);
            VMI = min(VF);
            [ri,ci,si] = size(V);
            P = imresize3(P,[ri,ci,si]);
            %             if exist('C:\Users\Administrator\Downloads\Labels\PET\'+string(ctfilelist(i).name), 'file')
            %                 PL = niftiread('C:\Users\Administrator\Downloads\Labels\PET\'+string(ctfilelist(i).name));
            %                 CL = niftiread('C:\Users\Administrator\Downloads\Labels\CT\'+string(ctfilelist(i).name));
            %                 PL =  imresize3(PL,[ri,ci,si]);
            %                 CL =  imresize3(CL,[ri,ci,si]);
            %                 niftiwrite(CL,newfolder+newfile+'ctlabel.nii');
            %                 niftiwrite(PL,newfolder+newfile+'petlabel.nii');
            %             end
            for s = 1:si
                V1 = uint8((double(squeeze(V(:,:,s)))/(VMA-VMI))*255);
                P1 = uint8((double(squeeze(P(:,:,s)))/(PMA-PMI))*255);

                % read one image set
                % a = 10;
                % setName = num2str(a);
                % imgVis = imread(strcat('vis\', setName, '.bmp'));
                % imgIR = imread(strcat('ir\', setName, '.bmp'));
                tic;
                % image fusion
                result = BGR_Fuse(V1, P1, QuadNormDim, QuadMinDim, GaussScale, MaxRatio, StdRatio);
                toc;
                % show image
                T(:,:,s) = result;
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