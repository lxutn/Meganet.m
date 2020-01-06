% Classification of the swiss roll data with a Hamiltonian ResNet
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

% generate training and validation data
[Ytrain,Ctrain,Yv,Cv] = setupSwissRoll(256);
%Ytrain=Ytrain+0.01*randn(size(Ytrain));

% sample = element in a row vector
Ctrain = Ctrain(1,:);
Cv = Cv(1,:);

minLevel = 4;
maxLevel = 6;

figure(1); clf;
subplot(2,10,1);
% visualize data with a 2d domain
viewFeatures2D(Ytrain,Ctrain)
axis equal tight
title('input features');
%% setup network
% Stop time for the evolution of the CT dynamics
T = 20;
% desired number of layers = number of sampling instants
nt = 2^minLevel;
% sampling time
h  = T/nt;
% extract the number of features = dimension of the domain = 2
nf = size(Ytrain,1);
% K and B are used for affine transformation of the data K*Y+b*B
K = dense([nf,nf]); % K does not store yet data
B = ones(nf,1);

% setup a Hamiltonian network  called net
net   = HamiltonianNN(@tanhActivation,K,B,nt,h);

%% setup classifier
pLoss = logRegressionLoss(); % compute the first part of (2.4) in stable DNN paper
%% setup regularizers
% My remark: the correct magnitude influences the results. It shoudl
% increase with the number of layers
%
% To be done for improving the results: tune it through cross validation

alpha = 5e-5;
regOp = opTimeDer(nTheta(net),nt,h); % xuliang: define a time difference operator, it's a matrix like [1, -1, 0; 0,1, -1], 
                                     % with which we can calculate the
                                     % difference between paarmeters, i.e.,
                                     % $K_i-K_{i+1}
pRegK = tikhonovReg(regOp,h*alpha,[]); % regularize K
regOpW = opEye((prod(sizeFeatOut(net))+1)*size(Ctrain,1)); % xuliang: identity matrix like object
pRegW = tikhonovReg(regOpW,1e-4); % regularize W
%% setup solver for classification problem
% xuliang: because it is block coordinate descent
classSolver = newton();
classSolver.maxIter=4;
classSolver.linSol.maxIter=3;

%% setup solver for outer minimization problem

% since the type of opt is newton, the newton solver will be used
opt = newton();
%opt = bfgs();
opt.out=2;
 opt.linSol.maxIter=20;
opt.atol = 1e-16;
 opt.linSol.tol=0.01;
opt.maxIter=100;
opt.LS.maxIter=20;

%% setup objective function for training and validation data

fctn = dnnVarProObjFctn(net,pRegK,pLoss,pRegW,classSolver,Ytrain,Ctrain); % liang: they include classSolve to elimite one variable W
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);

%% solve multilevel classification problem
% initialize the theta format of the parameters

theta = repmat([1;0;0;1;0],nt,1); % liang: vectorized version of [K, b]

for level=minLevel:maxLevel
    % setup preconditioner
    PCfun = @(x) PCmv(fctn.pRegTheta.B,x, alpha^2, net.h*1e-1);
    PC = LinearOperator(nTheta(fctn.net),nTheta(fctn.net),PCfun,PCfun);
     opt.linSol.PC = PC;
    
    % TRAIN the network
    % sovlve(netwon object, cost, init parameters, validation)
    
    [theta,his] = solve(opt,fctn,theta(:),fval);

    % plot the results
    [Jc,para] = eval(fctn,theta);
    WOpt = para.W;
        % compute the output withe the given weights
    [Yn,tmp] = forwardProp(net,theta,Yv);
    figure(1);
    subplot(2,maxLevel,level);
    viewFeatures2D(Yn,Cv);
    hold on;
        % smart rescaling of the axis
    ww = @(x) - (WOpt(1)*x + WOpt(3))./WOpt(2);
    ax = axis;
    plot(ax(1:2),ww(ax(1:2)),'-r')
    axis(ax);
    axis equal
    title('output features')
    axis equal tight
    subplot(2,maxLevel,level+maxLevel);
    viewContour2D([-1.2 1.2 -1.2 1.2],theta,reshape(WOpt,1,[]),net,pLoss);
    hold on
    viewFeatures2D(Yv,Cv);
    axis equal tight
    title('classification results')
    drawnow
    
    % create larger and larger DNN by doubling the layers at each time
    if level<maxLevel
        % prolongate weights to the next level
        % initialize the network with the double of the layers by
        % interpolating the weights found in the previous iteration
        
        % the output net of prolongateWeights has 2 times the layers of the
        % input net
        [net,theta] = prolongateWeights(fctn.net,theta);
        fctn.net = net;
        fval.net = net;
        fctn.pRegTheta.B     = opTimeDer(nTheta(net),net.nt,net.h);
        fctn.pRegTheta.alpha = net.h*alpha;
    end
end
