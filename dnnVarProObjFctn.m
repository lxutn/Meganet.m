classdef dnnVarProObjFctn < objFctn
    % classdef dnnVarObjFctn < objFctn
    %
    % variable projection objective function for deep neural networks 
    %
    % J(theta) = loss(h(W(theta)*Y(theta)), C) + Rtheta(Kb) + R(W(theta)),
    %
    % where W(theta) = argmin_W loss(h(W*Y(theta))), C) + R(W)
    
    % GIANNI's comments
    % The inner minimization is called "classification step" on pag 14 of
    % Ruthotto'' paper. It means to find the optimal weights in the
    % classification layer, given theta (the weights of the inner layers)
    % In othe words, W and theta are not optimized simultaneously but first
    % one finds theta and then W(theta). 
    
    properties
        net
        pRegTheta
        pLoss
        pRegW
        Y
        C
        matrixFree  % flag for matrix-free computation, default = 1
        gnHessian   % flag for Gauss-Newton approximation of Hessian
        optClass 
        useGPU      % flag for GPU computing
        precision   % flag for precision
    end
    
    methods
        function this = dnnVarProObjFctn(net,pRegTheta,pLoss,pRegW,optClass,Y,C,varargin)
            
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU    = [];
            precision = [];
            matrixFree = 1;
            gnHessian  = 1;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            
            this.net    = net;
            this.pRegTheta = pRegTheta;
            this.pLoss  = pLoss;
            this.pRegW  = pRegW;
            this.optClass = optClass;
            if not(isempty(useGPU))
                this.useGPU = useGPU;
            end
            if not(isempty(precision))
                this.precision=precision;
            end
            [Y,C] = gpuVar(this.useGPU,this.precision,Y,C);
            this.Y         = Y;
            this.C         = C;
            this.gnHessian = gnHessian;
            this.matrixFree = matrixFree;
            
        end
        
        function [Jc,para,dJ,H,PC] = eval(this,theta,idx)
            % the method eval allows one to obtain the value of the
            % objective function, the gradient and the Hessian when using a
            % specific minibatch.
            % Gradient and hessian are evaluated numerically
            if not(exist('idx','var')) || isempty(idx)
                Y = this.Y;
                C = this.C;
            else
                Y = this.Y(:,idx);
                C = this.C(:,idx);
            end
            compGrad = nargout>2;
            compHess = nargout>3;
            dJ = 0.0; H = []; PC = [];
            
            % project onto W 
            % NOT clear to me, even if W are the weights of the
            % classification layer 
            if compGrad || compHess
                % compute the DNN output of all layers for the current
                % theta and input data Y
                % tmp stores {Output Y of last layer, Output Z of last layer,
                % input Y to first layer (i.e. the data)}
                [YN,tmp] = forwardProp(this.net,theta,Y); % forward propagation
            else
                [YN,tmp] = forwardProp(this.net,theta,Y);
            end
            szYN  = size(YN);
            nex = szYN(end);
            YN = reshape(YN,[],nex);
            % creates fctn as the an object [classObjFctn} storing the 
            % objective function for classification i.e., 
            %
            %   J(W) = loss(h(W*Y), C) + R(W)
            %
            fctn  = classObjFctn(this.pLoss,this.pRegW,YN,C);
            % compute the optimal weight vector W, containing [W,mu]', of the classification layer
            % given the output data YN, the class of each datapoint C and the current theta
            W     = solve(this.optClass,fctn,zeros(size(C,1)*(size(YN,1)+this.pLoss.addBias),1,'like',theta));
            % compute dYF=dF/dY and d2YF=d2 F/d Y2
            [F,hisLoss,~,~,dYF,d2YF] = getMisfit(this.pLoss,W,YN,C); 
            dYF = reshape(dYF,szYN);
            if compGrad
                % compute dJ= d J/d theta
                % the size of dJ is indeed the same as the size of theta
                dJ = JthetaTmv(this.net,dYF,theta,Y,tmp);
            end
            % compute the Hessian matrix H= d2 J/d theta2
            % the next line is still dJ, as in the "if" above. why?
                dJ = JthetaTmv(this.net,dYF,theta,Y,tmp);
            if compHess
                if this.matrixFree
                    if this.gnHessian
                        % default case
                        HKbmv = @(x) JthetaTmv(this.net,reshape(d2YF*(Jthetamv(this.net,x,theta,Y,tmp)),szYN),theta,Y,tmp);
                        H   = LinearOperator(numel(theta),numel(theta),HKbmv,HKbmv);
                    else
                        HKbmv1 = @(x) JthetaTmv(this.net,reshape(d2YF*(Jthetamv(this.net,x,theta,Y,tmp)),szYN),theta,Y,tmp);
                        HKbmv2 = @(x) JthJthetaTmv(this.net,x,dYF,theta,Y,tmp);
                
                        HKbmv = @(x) HKbmv1(x)+HKbmv2(x);
                        H   = LinearOperator(numel(theta),numel(theta),HKbmv,HKbmv);
                    end
                else
                    % build the Hessian d2YF= dF2 /d Y2 (only possible in
                    % simple cases because it is a matrix in nex^2
                    % dimensions where nex is the number of datapoints !
                    switch class(this.pLoss)
                        case 'regressionLoss'
                            d2YF =(1/nex)* kron(speye(nex),W(:,1:end-this.pLoss.addBias)'*W(:,1:end-this.pLoss.addBias));
                        otherwise
                            error('matrix-based Hessians are not implemented for this loss function');
                    end

                    if this.gnHessian
                        H = getHessian(this.net,dYF,d2YF,theta,Y,tmp);
                    else
                        [HKb1,HKb2] = getHessian(this.net,dYF,d2YF,theta,Y,tmp);
                        H = HKb1 + HKb2;
                    end
                end
            end
            
            para = struct('F',F,'hisLoss',hisLoss);
            
            % ADD TO F, dF/d theta and dF2 /d theta2 the contribution of the
            % regularization term, i.e. \alpha*R, \alpha * d R/ d theta and
            % \alpha *
            % d2 R/ d theta2, respectively
            Jc   = F;
            
            % evaluate regularizer for DNN weights
            if not(isempty(this.pRegTheta))
                [Rth,hisRth,dRth,d2Rth]      = regularizer(this.pRegTheta,theta);
                Jc = Jc + Rth;
                if compGrad
                    dJ = dJ + dRth;
                end
                if compHess
                    H  = H + d2Rth;
                end
                para.Rth = Rth;
                para.hisRth = hisRth;
            end
            
            if not(isempty(this.pRegW))
                [RW,hisRW]         = regularizer(this.pRegW, W);
                Jc = Jc + RW;
                para.RW = RW;
                para.hisRW = hisRW;
            end
            para.W = W;
            if nargout>4
%                 PC = getPC(this.pRegTheta);
                 PC = []; 
                 %opEye(numel(theta)); % getPC(this.pRegTheta);
            end           
        end
        
        
        function [str,frmt] = hisNames(this)
            [str,frmt] = hisNames(this.pLoss);
            if not(isempty(this.pRegTheta))
                [s,f] = hisNames(this.pRegTheta);
                s{1} = [s{1} '(theta)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
            if not(isempty(this.pRegW))
                [s,f] = hisNames(this.pRegW);
                s{1} = [s{1} '(W)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
        end
        
        function his = hisVals(this,para)
            his = hisVals(this.pLoss,sum(para.hisLoss,1));
            if not(isempty(this.pRegTheta))
                his = [his, hisVals(this.pRegTheta,para.hisRth)];
            end
            if not(isempty(this.pRegW))
                his = [his, hisVals(this.pRegW,para.hisRW)];
            end
        end
        
        
        function str = objName(this)
            str = 'dnnVarPro';
        end
        
        function runMinimalExample(~)
            
            nex    = 400; nf =2;
            
            % create two DNN and concatenate them into a aingle DNN net
            blocks = cell(2,1);
            blocks{1} = NN({singleLayer(dense([2*nf nf]))});
            blocks{2} = ResNN(singleLayer(dense([2*nf 2*nf])),1,.1);
            net    = Meganet(blocks);
            nth = nTheta(net);
            theta  = randn(nth,1);
            
            Y = randn(nf,nex);
            C = zeros(nf,nex);
            C(1,Y(2,:)>Y(1,:).^2) = 1;
            C(2,Y(2,:)<=Y(1,:).^2) = 1;
            
            pLoss = softmaxLoss();
            W = vec(randn(2,2*nf+1));
            pRegW        = tikhonovReg(opEye(numel(W)));
            pRegTheta    = tikhonovReg(opEye(numel(theta)));
            
            newtInner =newton('out',0,'maxIter',5);
            % embed all useful information (DNN structure, loss function,
            % data) into a single object fctn
            fctn = dnnVarProObjFctn(net,pRegTheta,pLoss,pRegW,newtInner,Y,C);
            %[Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
            %  checkDerivative(fctn,Kb(:))
            % Create an object of type newton -> see newton.m 
            newtOuter =newton('out',1,'maxIter',60);
            % call the method solve in the object newtOuter 
            [KbWopt,his] = solve(newtOuter,fctn,theta(:));
        end
    end
end










