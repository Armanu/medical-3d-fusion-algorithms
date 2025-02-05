% 字典学习，K-SVD
function train_dictionary
V = niftiread('./HANCT.nii');
[ri,ci,si] = size(V);
disp(si)
disp(ci)
disp(ri)
P = niftiread('./HANPT.nii');
for inde=1:si
index = inde;
disp(num2str(index));
V1 = (double(squeeze(V(:,:,si)))/1000)*255;
P1 = (double(squeeze(P(:,:,si)))/4)*255;

I1 = im2double(V1);
I2 = im2double(P1);

[m,n] = size(I1);

unit = 7;
step = 3;
step2 = (unit-1)/2;

row_unit = unit*unit;

disp(strcat('开始计算分块数据'));
count = 0;
for i=(1+step2):step:(m-step2)
    for j=(1+step2):step:(n-step2)
        count = count+1;
        patch1 = I1((i-3):(i+3),(j-3):(j+3));
        patch2 = I2((i-3):(i+3),(j-3):(j+3));
        
        Vi1(:, count) = patch1(:);
        Vi2(:, count) = patch2(:);
    end
end
disp(strcat('结束计算分块数据'));

patch_num = count;
Vc = zeros(row_unit, patch_num*2);%联合数据

Vc(:,1:patch_num) = Vi1;
Vc(:,(patch_num+1):2*patch_num) = Vi2;

% KSVD训练子字典
dic_size = 256;
k = 16;

disp('KSVD-计算字典 D');
tic
params.data = Vc;
params.Tdata = k;
params.dictsize = dic_size;
params.iternum = 50;
params.memusage = 'high';
[D,X,err] = ksvd(params,'');
toc
disp('KSVD-计算字典 结束');

%保存字典
D_path = './dictionary/';
D_name = [D_path, 'D_unit7_im',num2str(index),'.dat'];
save(D_name,'D');

clear Vi1;
clear Vi2;
end
% T(:,:,si) = fusion;
% end
% niftiwrite(T,'outbrain.nii');








