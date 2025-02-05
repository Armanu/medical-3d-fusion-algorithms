%    The following is an implementation of an infrared and visible image
%    fusion algorithm proposed in the paper:
%
%      "Perceptual fusion of infrared and visible images through a hybrid
%      multi-scale decomposition with Gaussian and bilateral filters"
%      Information Fusion, 2016.
%
%    This code is for testing purpose only.
%    Some of the test images were obtained at
%      http://www.imagefusion.org
%      http://www.ece.lehigh.edu/SPCRL/IF/image_fusion.htm
%
%    Zhiqiang Zhou, Beijing Institute of Technology
%    May, 2015

clear all;
close all;

nLevel = 4;
lambda = 30;
% lambda = 3000;

% path_Vis = '.\image\Camp_Vis.jpg';      path_IR = '.\image\Camp_IR.jpg';
% path_Vis = '.\image\Trees4917_Vis.jpg'; path_IR = '.\image\Trees4917_IR.jpg';
% path_Vis = '.\image\Octec_Vis.jpg';     path_IR = '.\image\Octec_IR.jpg';
% path_Vis = '.\image\Road_Vis.jpg';      path_IR = '.\image\Road_IR.jpg';
%path_Vis = './ir/10.bmp'; path_IR = './vis/10.bmp';
% path_Vis = '.\image\Kayak_Vis.jpg';     path_IR = '.\image\Kayak_IR.jpg';
% path_Vis = '.\image\Steamboat_Vis.jpg'; path_IR = '.\image\Steamboat_IR.jpg';


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
    newfolder = 'G:\HYBRID-MSD\'+newfile+'\';
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

%                 V1 = double(squeeze(V(:,:,s)));
%                 V2 = V1;
%                 for r = 1:ri
%                     for c = 1:ci
%                         res = round(abs(((V1(r,c)+(VMA-VMI))/(2*(VMA-VMI)))*255));
%                         if (res > 255)
%                             V2(r,c) = 255;
%                         elseif (res < 0)
%                             V2(r,c) = 0;
%                         else
%                             V2(r,c) = res;
%                         end
%                     end
%                 end
% 
% 
%                 P1 = double(squeeze(P(:,:,s)));
%                 P2 = P1;
%                 for pr = 1:ri
%                     for pc = 1:ci
%                         pres = round(abs((P1(pr,pc)/(PMA-PMI))*255));
%                         if (pres > 255)
%                             P2(pr,pc) = 255;
%                         elseif (pres < 0)
%                             P2(pr,pc) = 0;
%                         else
%                             P2(pr,pc) = pres;
%                         end
%                         %disp(P2(pr,pc));
%                     end
%                 end

                % [img1, img2, para.name] = PickName(path_Vis, path_IR);
                paraShow1.fig = 'Visible image';
                paraShow2.fig = 'Infrared image';
                %% ---------- Hybrid Multi-scale Decomposition --------------
                sigma = 2.0;
                sigma_r = 0.05;
                k = 2;

                M1 = cell(1, nLevel+1);
                M1L = cell(1, nLevel+1);
                M1{1} = V2;
                M1L{1} = V2;
                M1D = cell(1, nLevel+1);
                M1E = cell(1, nLevel+1);
                sigma0 = sigma;
                for j = 2:nLevel+1
                    w = floor(3*sigma0);
                    h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);
                    M1{j} = imfilter(M1{j-1}, h, 'symmetric');
                    %M1L{j} = 255*bfilter2(M1L{j-1}/255,w,[sigma0, sigma_r/(k^(j-2))]);
                    M1L{j} = 255*fast_bfilter2(M1L{j-1}/255,[sigma0, sigma_r/(k^(j-2))]);

                    M1D{j} = M1{j-1} - M1L{j};
                    M1E{j} = M1L{j} - M1{j};

                    sigma0 = k*sigma0;
                end

                M2 = cell(1, nLevel+1);
                M2L = cell(1, nLevel+1);
                M2{1} = P2;
                M2L{1} = P2;
                M2D = cell(1, nLevel+1);
                M2E = cell(1, nLevel+1);
                sigma0 = sigma;
                for j = 2:nLevel+1
                    w = floor(3*sigma0);
                    h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);
                    M2{j} = imfilter(M2{j-1}, h, 'symmetric');
                    %M2L{j} = 255*bfilter2(M2L{j-1}/255,w,[sigma0, sigma_r/(k^(j-2))]);
                    M2L{j} = 255*fast_bfilter2(M2L{j-1}/255,[sigma0, sigma_r/(k^(j-2))]);

                    M2D{j} = M2{j-1} - M2L{j};
                    M2E{j} = M2L{j} - M2{j};

                    sigma0 = k*sigma0;
                end

                %% ---------- Multi-scale Combination --------------
                for j = nLevel+1:-1:3
                    b2 = abs(M2E{j});
                    b1 = abs(M1E{j});
                    R_j = max(b2-b1, 0);
                    Emax = max(R_j(:));
                    P_j = R_j/Emax;

                    C_j = atan(lambda*P_j)/atan(lambda);

                    % Base level combination
                    sigma0 = 2*sigma0;
                    if j == nLevel+1
                        w = floor(3*sigma0);
                        h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);
                        lambda_Base = lambda;
                        %lambda_Base = 30;
                        C_N = atan(lambda_Base*P_j)/atan(lambda_Base);
                        C_N = imfilter(C_N, h, 'symmetric');
                        MF = C_N.*M2{j} + (1-C_N).*M1{j};
                    end

                    % Large-scale combination
                    sigma0 = 1.0;
                    w = floor(3*sigma0);
                    h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);
                    C_j = imfilter(C_j, h, 'symmetric');

                    D_F = C_j.*M2E{j}+ (1-C_j).*M1E{j};
                    MF = MF + D_F;
                    D_F = C_j.*M2D{j}+ (1-C_j).*M1D{j};
                    MF = MF + D_F;
                end

                % % Small-scale combination
                % sigma0 = 0.2;
                % w = floor(3*sigma0);
                % h = fspecial('gaussian', [2*w+1, 2*w+1], sigma0);
                % C_0 = double(abs(M1E{2}) < abs(M2E{2}));
                % C_0 = imfilter(C_0, h, 'symmetric');
                % D_F = C_0.*M2E{2} + (1-C_0).*M1E{2};
                % MF = MF + D_F;
                % C_0 = abs(M1D{2}) < abs(M2D{2});
                % C_0 = imfilter(C_0, h, 'symmetric');
                % D_F = C_0.*M2D{2} + (1-C_0).*M1D{2};
                % MF = MF + D_F;

                %% ---------- Fusion Result --------------
                %FI = ImRegular(MF);   % The intensities are regulated into [0, 255]
                FI = max(min(MF,255), 0);
                %paraShow.fig = 'Fusion result';
                %ShowImageGrad(MF, paraShow);
                T(:,:,s) = MF;
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

