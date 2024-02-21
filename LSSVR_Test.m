%% LSSVR建模测试样例 || test case of LSSVR modeling
% 结果：e_lssvr = 0.0462 （RMSE）；time_lssvr = 0.01651 （秒）
% results: e_lssvr = 0.0462 (RMSE); time_lssvr = 0.01651 s
%% matlab初始化 || matlab initialization
clc
clear
close all
% 定义一个随时间变化的初值，用于产生不同的随机数
% define an initial value that varies over time to generate different
% random data
rand('state', sum(100 * clock));
%% 训练样本800条，测试样本200条，比例4：1  || training data 800 and test data 200
% 设计多输入单输出非线性测试函数
% 行为数据样本数量，列为数据样本维度
% 数据样本没有加入噪声
% rand函数自动产生(0,1)之间的随机数，不需要进行数据归一化操作
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
%% 画出真实曲线图 || draw the actual graph
datanum = 1:Ntest;
plot(datanum, Ytest, 'k-', 'linewidth', 1.5);
hold on;
%% 调用LSSVR函数 || call the LSSVR function
type = 'function estimation';
% train the LSSVR model
[gam,sig2] = tunelssvm({X,Y,type,[],[],'RBF_kernel'},'simplex','leaveoneoutlssvm',{'mse'});
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'});
tic;
% make the prediction with the LSSVR model
lssvr_test = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
time_lssvr = toc;  % 记录预测计算时间
%% 给出预测结果 || record the predictive value
% 以RMSE为指标，给出偏差结果
e_lssvr = sqrt(mse(lssvr_test - Ytest));
% 画图与真实曲线做对比
% compare the graph with the actual curve
plot(datanum,lssvr_test,'ko','linewidth',1.5);
hold on;
legend('真实值','LSSVR预测值'); 
xlabel('测试数据编号'),ylabel('输出值');
hold off;