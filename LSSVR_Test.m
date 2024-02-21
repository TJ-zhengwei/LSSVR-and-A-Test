%% LSSVR��ģ�������� || test case of LSSVR modeling
% �����e_lssvr = 0.0462 ��RMSE����time_lssvr = 0.01651 ���룩
% results: e_lssvr = 0.0462 (RMSE); time_lssvr = 0.01651 s
%% matlab��ʼ�� || matlab initialization
clc
clear
close all
% ����һ����ʱ��仯�ĳ�ֵ�����ڲ�����ͬ�������
% define an initial value that varies over time to generate different
% random data
rand('state', sum(100 * clock));
%% ѵ������800������������200��������4��1  || training data 800 and test data 200
% ��ƶ����뵥��������Բ��Ժ���
% ��Ϊ����������������Ϊ��������ά��
% ��������û�м�������
% rand�����Զ�����(0,1)֮��������������Ҫ�������ݹ�һ������
% the multi-input single-output nonlinear test function is designed
Ntrain = 800;
x1 = rand(Ntrain, 1);
x2 = rand(Ntrain, 1);
x3 = rand(Ntrain, 1);
x4 = rand(Ntrain, 1);
X = [x1 x2 x3 x4];
Y = exp((x1.^2 + x2.^2) * 2) + sin(x3) + 2 * x4;
%-------------------------------------------------------------%
Ntest = 200;
xtest1 = rand(Ntest, 1);
xtest2 = rand(Ntest, 1);
xtest3 = rand(Ntest, 1);
xtest4 = rand(Ntest, 1);
Xtest = [xtest1 xtest2 xtest3 xtest4];
Ytest = exp((xtest1.^2 + xtest2.^2) * 2) + sin(xtest3) + 2 * xtest4;
%% ������ʵ����ͼ || draw the actual graph
datanum = 1:Ntest;
plot(datanum, Ytest, 'k-', 'linewidth', 1.5);
hold on;
%% ����LSSVR���� || call the LSSVR function
type = 'function estimation';
% train the LSSVR model
[gam,sig2] = tunelssvm({X,Y,type,[],[],'RBF_kernel'},'simplex','leaveoneoutlssvm',{'mse'});
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'});
tic;
% make the prediction with the LSSVR model
lssvr_test = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
time_lssvr = toc;  % ��¼Ԥ�����ʱ��
%% ����Ԥ���� || record the predictive value
% ��RMSEΪָ�꣬����ƫ����
e_lssvr = sqrt(mse(lssvr_test - Ytest));
% ��ͼ����ʵ�������Ա�
% compare the graph with the actual curve
plot(datanum,lssvr_test,'ko','linewidth',1.5);
hold on;
legend('��ʵֵ','LSSVRԤ��ֵ'); 
xlabel('�������ݱ��'),ylabel('���ֵ');
hold off;