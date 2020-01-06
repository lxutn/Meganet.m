classdef classObjFctn < objFctn
    % classdef classObjFctn < objFctn
    %
    % Objective function for classification,i.e., 
    %
    %   J(W) = loss(h(W*Y), C) + R(W),
    %
    % where 
    % 
    %   W    - weights of the classifier
    %   h    - hypothesis function
    %   Y    - features
    %   C    - class labels
    %   loss - loss function object
    %   R    - regularizer (object)
    
    properties
        pLoss
        pRegW
        Y
        C
    end
    
    methods
        function this = classObjFctn(pLoss,pRegW,Y,C)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            
            this.pLoss  = pLoss;
            this.pRegW  = pRegW;
            this.Y      = Y;
            this.C      = C;
            
        end
        
        function [Jc,para,dJ,H,PC] = eval(this,W,idx)
            % OUTPUT ARGUMENTS
            % Jc: cost 1/s*S(h(W*Y+\mu),C)+\alpha * R(W,mu,K...,B...) 
            % (see (2.3) in ruthotto's paper) 
            
            % dJ= gradient of Jc w.r.t the classification weights [vec(W);mu] 
            
            % H: hessian of Jc w.r.t the classification weights [vec(W);mu]
            
            if not(exist('idx','var')) || isempty(idx)
                Y = this.Y;
                C = this.C;
            else
                Y = this.Y(:,idx);
                C = this.C(:,idx);
            end
            % compute here the gradient and hessian of the part of the cost F= 1/s*S(h(W*Y+\mu),C) 
            % (see (2.3) in ruthotto's paper) w.r.t the
            % classification weights [vec(W);mu] by calling the method getMisfit for
            % the object ths.pLoss of, e.g. [logRegressionLoss] type
            % 
            % Jc is the classification cost 1/s*S(h(W*Y+\mu),C) that is the first part of the cost
            % in (2.3) in ruthotto's paper
            % hisloss = [F*s,s, {(# misclassified 0)+(# misclassified 1)}/2];
            % dJ, and H are the jacobian and hessian d F/ dW and d2 F/d W2
            
            [Jc,hisLoss,dJ,H] = getMisfit(this.pLoss,W,Y,C);
            para = struct('F',Jc,'hisLoss',hisLoss);
            
            
            if not(isempty(this.pRegW))
                % if there is a regularization component in the cost,
                % compute the jacobian and the hessian of thsi part wrt the
                % classification weights
                
                % compute here the gradient and hessian of the part of the cost Rc=\alpha * R(W,mu,K...,B...) 
            % (see (2.3) in ruthotto's paper) w.r.t the
            % classification weights [vec(W);mu] by calling the method regularizer for
            % the object ths.pRegW of, e.g. [tikhonovReg] type
                [Rc,hisReg,dR,d2R] = regularizer(this.pRegW,W);
                para.hisReg = hisReg;
                para.Rc     = Rc;
                % since J=1/s*S(h(W*Y+\mu),C)+R(W,mu), then dJ=d{1/s*S(h(W*Y+\mu),C)} +dR(W,mu)
                % same for the hessian matrix
                Jc = Jc + Rc; 
                dJ = vec(dJ)+ vec(dR);
                H = H + d2R;
                para.hisRW = hisReg;
            end

            if nargout>4
%                 PC = opEye(numel(W));
                PC = getPC(this.pRegW);
%                 PC = @(x) d2R\x;
            end
        end
        
        function [str,frmt] = hisNames(this)
            [str,frmt] = hisNames(this.pLoss);
            if not(isempty(this.pRegW))
                [s,f] = hisNames(this.pRegW);
                s{1} = [s{1} '(W)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
        end
        
        function his = hisVals(this,para)
            his = hisVals(this.pLoss,para.hisLoss);
            if not(isempty(this.pRegW))
                his = [his, hisVals(this.pRegW,para.hisRW)];
            end
        end
        
        
        function str = objName(this)
            str = 'classObjFun';
        end
        
        function runMinimalExample(~)
            
            pClass = regressionLoss();
            
            nex = 400;
            nf  = 2;
            nc  = 2;
            
            Y = randn(nf,nex);
            C = zeros(nf,nex);
            
            C(1,Y(2,:)>0) = 1;
            C(2,Y(2,:)<=0) = 1;
            
            W = vec(randn(nc,nf+1));
            pReg   = tikhonovReg(speye(numel(W)));
            
            fctn = classObjFctn(pClass,pReg,Y,C);
            opt1  = newton('out',1,'maxIter',20);
            
%             checkDerivative(fctn,W);
            [Wopt,his] = solve(opt1,fctn,W);
        end
    end
end










