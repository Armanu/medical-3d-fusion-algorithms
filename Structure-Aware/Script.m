clear;
close all
addpath(genpath(pwd));
%%----------------START OF NIFTY FOLDER LOOP-----------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
ctfilelist = dir(fullfile('C:\Users\Administrator\Downloads\CT\CT\', '*.nii.gz'));

for i = 1:length(ctfilelist)
    newfile = split(string(ctfilelist(i).name),'.');
    newfile = string(newfile{1});
    newfolder = 'C:\Users\Administrator\Documents\GitHub\medical-3d-fusion-algorithms\Structure-Aware\OUTPUT\'+newfile+'\';
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
                V1 = (double(squeeze(V(:,:,s)))/(VMA-VMI))*255;
                P1 = (double(squeeze(P(:,:,s)))/(PMA-PMI))*255;

                %     J(:,:,1) = V1;
                %     J(:,:,2) = P1;
                F_echo_dtf = IJF(V1,P1);
                %imwrite(F_echo_dtf,[num2str(i),'.bmp']);

                T(:,:,s) = F_echo_dtf;
                clear J;
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

%Q_echo_dtf = Qp_ABF(V1, P1, F_echo_dtf)
