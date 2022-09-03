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


ctfilelist = dir(fullfile('C:\Users\Administrator\Documents\Normalized\CT\', '*.nii'));

for i = 1:length(ctfilelist)
    newfile = split(string(ctfilelist(i).name),'.');
    newfile = string(newfile{1});
    newfolderct = 'G:\Croped\CT\'+newfile+'\';
    newfolderpet = 'G:\Croped\PET\'+newfile+'\';
    
    if ~exist(newfolderct, 'dir')
        if exist('C:\Users\Administrator\Documents\Normalized\PET\'+string(ctfilelist(i).name), 'file')
            V = niftiread('C:\Users\Administrator\Documents\Normalized\CT\'+string(ctfilelist(i).name));
            P = niftiread('C:\Users\Administrator\Documents\Normalized\PET\'+string(ctfilelist(i).name));
            clear T;
            clear PT;
            
            if ~exist(newfolderct,'dir')
                mkdir(newfolderct)
                mkdir(newfolderpet)
                
            end
            
            disp("file : "+ string(ctfilelist(i).name))
            
            
            
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
            fi = uint8(si-(si/8.5));
            st = uint8((si*3)/8.5);
            for s = st:fi
                disp(string(s))
                
                
                V1 = squeeze(V(:,:,s));
                P1 = squeeze(P(:,:,s));
                
                % read one image set
                % a = 10;
                % setName = num2str(a);
                % imgVis = imread(strcat('vis\', setName, '.bmp'));
                % imgIR = imread(strcat('ir\', setName, '.bmp'));
                
                % image fusion
                %result = BGR_Fuse(V1, P1, QuadNormDim, QuadMinDim, GaussScale, MaxRatio, StdRatio);
                
                % show image
                T(:,:,s) = V1;
                PT(:,:,s) = P1;
                
            end
            
            niftiwrite(T,newfolderct+newfile+'.nii.gz');
            niftiwrite(PT,newfolderpet+newfile+'.nii.gz');
            
        end
    else
        disp("been there");
    end
end
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------END OF NIFTY FOLDER LOOP------------------%%