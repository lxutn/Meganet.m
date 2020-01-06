classdef logRegressionLoss
    % classdef logRegressionLoss
    %
    % object describing logistic regression loss function
    
    properties
        theta
        addBias
    end
    
    
    methods
        function this = logRegressionLoss(varargin)
            this.theta   = 1e-3;
            this.addBias = 1;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        
        
        function [F,para,dWF,d2WF,dYF,d2YF] = getMisfit(this,W,Y,C,varargin)
            % OUTPUTS
            
            % F is 1/s*S(h(W*Y+\mu),C) that is the first part of the cost
            % in (2.4) in ruthotto's paper
            
            % dWF, and d2WF are the jacobian and hessian d F/ dW and d2 F/d W2
            % dYF, and d2YF are the jacobian and hessian d F/ dY and d2 F/d Y2
            
            % para = [F*1/s,s, {(# misclassified 0)+(# misclassified 1)}/2]; 
            doDY = (nargout>3);
            doDW = (nargout>1);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            dWF = []; d2WF = []; dYF =[]; d2YF = [];
            
            szY  = size(Y);
            nex  = szY(2); % number of datapoints in Y
            if this.addBias==1
                Y   = [Y; ones(1,nex)];
            end
            szW  = [size(C,1),size(Y,1)]; % set the size of the weight matrix W, which includes also the bias \mu in the last element
             W    = reshape(W,szW);
            
            % computation of the function S(h(W*Y+\mu),C) and of its derivatives w.r.t [vec(W);mu], see (2.3) in ruthotto's paper
            S  = W*Y;
            Cp   = getLabels(this,S);
            err  = nnz(C-Cp)/2; % err is the half of the number of ?misclassified? datapoints
            % find the misclassified datapoints
            posInd = (S>0);
            negInd = (0 >= S);
            
            % compute the loss S and store it in F. S is J(\theta) in my
            % notes about logistic regression and ML
            F = -sum(C(negInd).*S(negInd) - log(1+exp(S(negInd))) ) - ...
                sum(C(posInd).*S(posInd) - log(exp(-S(posInd))+1) - S(posInd));
            para = [F,nex,err]; % nex is the number of datapoints
            F  = F/nex; % divide the cost by the number of ?misclassified? datapoint
            
            % Compute the jacobian d F/d theta
            if (doDW) && (nargout>=2)
                dF  = (C - 1./(1+exp(-S)));
                dWF = -Y*dF';
                dWF = vec(dWF)/nex;
            end
            
            % Compute the Hessian d2 F/d theta2
            if (doDW) && (nargout>=3)
                d2F  = 1./(2*cosh(S/2)).^2;
                matW  = @(W) reshape(W,szW);
                d2WFmv = @(U) Y*(((d2F + this.theta).*(matW(U/nex)*Y)))';
%                 d2WFmv = @(U) (((d2F + this.theta).*(matW(U/nex)*Y)))*Y';
                d2WF = LinearOperator(prod(szW),prod(szW),d2WFmv,d2WFmv);
            end
            if doDY && (nargout>=4)
                if this.addBias==1
                    W = W(:,1:end-1);
                end
                dYF  =   -vec(W'*dF)/nex;
            end
            if doDY && nargout>=5
                WI     = @(T) W*T;  %kron(W,speye(size(Y,1)));
                WIT    = @(T) W'*T;
                matY   = @(Y) reshape(Y,szY);
%                  d2YFmv = @(T) vec(WIT(((d2F(WI(matY(T/nex)))))));
                d2YFmv = @(T) WIT((d2F + this.theta).*(WI(matY(T/nex))));
    
                d2YF = LinearOperator(prod(szY),prod(szY),d2YFmv,d2YFmv);
            end
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'F','accuracy'};
            frmt = {'%-12.2e','%-12.2f'};
        end
        function str = hisVals(this,para)
            str = [para(1)/para(2),(1-para(3)/para(2))*100];
        end
        function [Cp,P] = getLabels(this,W,Y)
            % compute, for each sample y_i in Y
            % S_i=(W*y_i + mu)
            % P_i = probability that S_i = 1 (double check this ...)
            % P_i= 1/(1+e^-S_i) if S_i>0 or
            % P_i= 1-1/(1+e^-S_i)=... if S_i<0
            % Cp_i=1 if and only if P_i>0.5
            if nargin==2
                S = W;
            else
                [nf,nex] = size(Y);
                W      = reshape(W,[],nf+1);
                if this.addBias==1
                    Y     = [Y; ones(1,nex)];
                end
                
                
                S      = W*Y;
            end
            posInd = (S>0);
            negInd = (0 >= S);
            P = 0*S;
            P(posInd) = 1./(1+exp(-S(posInd)));
            P(negInd) = exp(S(negInd))./(1+exp(S(negInd)));
                Cp = P>.5;
            
        end
    end
    
end

