% Classification of the swiss roll data with an antisymmetric ResNet
%
% This example is similar to the one described in Sec. 6.2 of
% 
% @article{HaberRuthotto2017,
%   author = {Haber, Eldad and Ruthotto, Lars},
%   title = {{Stable architectures for deep neural networks}},
%   journal = {Inverse Problems},
%   year = {2017},
%   volume = {34},
%   number = {1},
%   pages = {1--22},
% }

clear all; close; 

[Ytrain,Ctrain,Yv,Cv] = setupSwissRoll(256);
%Ytrain=Ytrain+0.01*randn(size(Ytrain));
Ytrain = [Ytrain; zeros(2,size(Ytrain,2))];
Yv = [Yv; zeros(2,size(Yv,2))];

Ctrain = Ctrain(1,:);
Cv = Cv(1,:);

minLevel = 4;
maxLevel = 4;
%maxLevel = 10;

figure(1); clf;
subplot(2,10,1);
viewFeatures2D(Ytrain,Ctrain)
axis equal tight
title('input features');
%% setup networks
T = 20;
nt = 2^minLevel;
h  = T/nt;
nf = size(Ytrain,1);

K     = getDenseAntiSym([nf,nf]);
% define a single layer Y(th,Y0) = activation( K(th_1)*Y0 + Bin*th_2)+Bout*th_3
layer = singleLayer(K,'Bin',ones(nf,1));
net   = ResNN(layer,nt,h);

%% setup classifier
pLoss = logRegressionLoss();
%% setup regularizer
alpha = 5e-4;
% set the matrix that computes difference of weights across consecutive layers 
regOp = opTimeDer(nTheta(net),nt,h);
pRegK = tikhonovReg(regOp,h*alpha,[]);
% set the identity matrix for minimizing ||W||^2
regOpW = opEye((prod(sizeFeatOut(net))+1)*size(Ctrain,1));
pRegW = tikhonovReg(regOpW,1e-3);
%% setup solver for classification problem
classSolver = newton();
classSolver.maxIter=4;
classSolver.linSol.maxIter=3;

%% setup solver for outer optimization problem
opt = newton();
opt.out=2;
opt.linSol.maxIter=20;
opt.atol = 1e-16;
opt.linSol.tol=0.01;
opt.maxIter=100;
opt.LS.maxIter=10;

%% define objective functions for training and validation data
fctn = dnnVarProObjFctn(net,pRegK,pLoss,pRegW,classSolver,Ytrain,Ctrain);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

%% do multilevel training
theta = 0*initTheta(net);
for level=minLevel:maxLevel
    % setup preconditioner
    PCfun = @(x) PCmv(fctn.pRegTheta.B,x, alpha^2, net.h*1e-1);
    PC = LinearOperator(nTheta(fctn.net),nTheta(fctn.net),PCfun,PCfun);
    opt.linSol.PC = PC;
    
    % solve the optimization problem
    [theta,his] = solve(opt,fctn,theta(:),fval);

    % plot the results
    [Jc,para] = eval(fctn,theta);
    WOpt = para.W;
    % propagate the validation datapoints
    [Yn,tmp] = forwardProp(net,theta,Yv);
    figure(1);
    subplot(2,10,level);
    viewFeatures2D(Yn,Cv);
    axis equal
    title('output features')
    axis equal tight
    subplot(2,10,level+10);
    viewContour2D([-1.2 1.2 -1.2 1.2],theta,reshape(WOpt,1,[]),net,pLoss);
    hold on
    viewFeatures2D(Yv,Cv);
    axis equal tight
    title('classification results')
    drawnow
    
    if level<maxLevel
        % prolongate weights to the next level
        [net,theta] = prolongateWeights(fctn.net,theta);
        fctn.net = net;
        fval.net = net;
        fctn.pRegTheta.B     = opTimeDer(nTheta(net),net.nt,net.h);
        fctn.pRegTheta.alpha = net.h*alpha;
    end
end