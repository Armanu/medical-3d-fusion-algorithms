%%% Image Fusion using Details computed from Cross Bilteral Filter output.
%%% Details are obtained by subtracting original image by cross bilateral filter output.
%%% These details are used to find weights (Edge Strength) for fusing the images.
%%% Author : B. K. SHREYAMSHA KUMAR

%%% Copyright (c) 2013 B. K. Shreyamsha Kumar
%%% All rights reserved.

%%% Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy,
%%% modify, and distribute this code (the source files) and its documentation for any purpose, provided that the
%%% copyright notice in its entirety appear in all copies of this code, and the original source of this code,
%%% This should be acknowledged in any publication that reports research using this code. The research is to be
%%% cited in the bibliography as:

%%% B. K. Shreyamsha Kumar, image fusion based on pixel significance using cross bilateral filter",
%%% Signal, Image and Video Processing, pp. 1-12, 2013. (doi: 10.1007/s11760-013-0556-9)

%%% Fusion Method Parameters.
cov_wsize=5;

%%% Bilateral Filter Parameters.
sigmas=1.8;  %%% Spatial (Geometric) Sigma. 1.8
sigmar=25; %%% Range (Photometric/Radiometric) Sigma.25 256/10
ksize=11;   %%% Kernal Size  (should be odd).


%%----------------START OF NIFTY FOLDER LOOP-----------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%
%%-----------------------------------------------------------%%

ctfilelist = dir(fullfile('C:\Users\Administrator\Downloads\CT\CT\', '*.nii.gz'));

for i = 1:length(ctfilelist)
    newfile = split(string(ctfilelist(i).name),'.');
    newfile = string(newfile{1});
    newfolder = 'C:\Users\Administrator\Documents\GitHub\medical-3d-fusion-algorithms\CBF\OUTPUT\'+newfile+'\';
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
                
                x{1}=V1;
                x{2}=P1;
                
                [M,N]=size(x{1});
                
                %%% Cross Bilateral Filter.
                disp("slice : "+ string(s))
                
                cbf_out{1}=cross_bilateral_filt2Df(x{1},x{2},sigmas,sigmar,ksize);
                detail{1}=double(x{1})-cbf_out{1};
                cbf_out{2}= cross_bilateral_filt2Df(x{2},x{1},sigmas,sigmar,ksize);
                detail{2}=double(x{2})-cbf_out{2};
                
                %%% Fusion Rule (IEEE Conf 2011).
                xfused=cbf_ieeeconf2011f(x,detail,cov_wsize);
                                
                xfused8=uint8(xfused);
                % figure,imshow(xfused8);
                
                % imwrite(xfused8,fused_path,'png');
                
                % if(strncmp(inp_image,'gun',3))
                %    figure,imagesc(x{1}),colormap gray
                %    figure,imagesc(x{2}),colormap gray
                %    figure,imagesc(xfused8),colormap gray
                % else
                %    figure,imshow(x{1})
                %    figure,imshow(x{2})
                %    figure,imshow(xfused8)
                % end
                
                % axis([140 239 70 169]) %%% Office.
                
                % fusion_perform_fn(xfused8,x);
                
                T(:,:,s) = xfused8;
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

% end