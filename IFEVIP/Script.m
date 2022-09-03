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


ctfilelist = dir(fullfile('G:Croped\CT\*\', '*.nii'));

for i = 1:length(ctfilelist)
    newfile = split(string(ctfilelist(i).name),'.');
    newfile = string(newfile{1});
    %disp(newfile)
    newfolder = 'G:\IFEVIP\'+newfile+'\';
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

                % read one image set
                % a = 10;
                % setName = num2str(a);
                % imgVis = imread(strcat('vis\', setName, '.bmp'));
                % imgIR = imread(strcat('ir\', setName, '.bmp'));
                %tic;
                % image fusion
                result = BGR_Fuse(V1, P1, QuadNormDim, QuadMinDim, GaussScale, MaxRatio, StdRatio);
                %toc;
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