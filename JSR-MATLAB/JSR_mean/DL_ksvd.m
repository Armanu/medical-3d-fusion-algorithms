%% ʹ��KSVD�ֵ�ѧϰ�㷨��ѧϰ���ֵ�D����СΪ64*256
% ͼ��ʹ��USC-SIPI ���ݼ�

path_file = './USC_SIPI_database/';

path_name = [path_file,'*.tiff'];
fileP=dir(path_name);
fileCount = length(fileP); % �ļ�����
% ͼ����СΪ8
unit = 8;
v_num = 50;

V = zeros(unit*unit, v_num*fileCount);
disp(strcat('��ʼ'));
for index = 1:fileCount
    str = ['index = ',num2str(index)];
    disp(strcat(str));
    path = [path_file,fileP(index).name];
    I = imread(path);
    d = size(I,3);
    if d>1
        I = rgb2gray(I);
    end
    I = im2double(I);
    Vi = patchVector(I, unit, v_num);
    V(:, ((index-1)*v_num+1):(index*v_num)) = Vi;
end
save('V_image_patch.dat','V');

dic_size = 256;
k=5;
disp('KSVD-�����ֵ� ��ʼ');
params.data = V;
params.Tdata = k;
params.dictsize = dic_size;
params.iternum = 50;
params.memusage = 'high';
[D,X,err] = ksvd(params,'');
disp('KSVD-�����ֵ� ����');

save('D_k5_ksvd.dat','D');











