classdef newton < optimizer
    % classdef newton < optimizer
    %
    % Newton-CG method for minimizing non-linear objective function. 
    % 
    % To use this code, the objective function must be of type objFctn or a 
    % function handle that can be called like
    %
    % [Jc,para,dJ,H,PC] = fctn(xc);
    %
    % where 
    %   Jc   - function value gradient 
    %   para - struct containing information about the objectcive used for
    %          plotting or printing (see, e.g., method hisVals in objFctn)
    %   dJ   - gradient about xc
    %   H    - Hessian approximation
    %   PC   - preconditioner or empty
    %   
    % using these ingredients a search direction is computed using a PCG
    % solver, specified by the property linSol and a line search is
    % performed. 
    %
    % The property 'out' controls the verbosity of the method. If out==1
    % each iteration will produce some output in the command window. There
    % will be a table containing the following columns:
    %
    % iter - current iteration
    % Jc   - objective function value
    % |x-xOld| - size of step taken in this iteration
    % |dJ|/|dJ0| - relative norm of gradient
    % iterCG     - number of PCG iterations
    % relresCG   - relative residual achieved by PCG
    % mu         - line search parameter
    % LS         - number of line search steps
    %
    % When the objective function is a subtype of 'objFctn' there will be
    % additional output specified by the respective class. For example, the
    % dnnObjFctn will produce the following output
    %
    % F          - loss
    % accuracy   - classification accuracy
    % R(theta)   - value of regularizer for network paramters
    % alpha      - regularization parameter for theta 
    % R(W)       - value of regularizer for classfication weights
    
    properties
        maxIter   % maximum number of iterations
        atol      % absolute tolerance for stopping, stop if norm(dJ)<atol
        rtol      % relative tolerance for stopping, stop if norm(dJ)/norm(dJ0)<rtol
        maxStep   % maximum step (similar to trust region methods)
        out       % flag controlling the verbosity, 
                  %       out==0 -> no output
                  %       out==1 -> print status at each iteration
        LS        % line search object. See, e.g., Armijo.m
        linSol    % linear solver object. See, e.g., steihaugPCG
    end
    
    methods
        
        function this = newton(varargin)
            % constructor, no input required to get instance with default
            % parameters
            this.maxIter = 10;
            this.atol    = 1e-3;
            this.rtol    = 1e-3;
            this.maxStep = 1.0;
            this.out     = 0;
            this.linSol  = steihaugPCG('tol',1e-1,'maxIter',10);
            this.LS      = Armijo();
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end;
        end
        
        function [str,frmt] = hisNames(this)
            % define the labels for each column in his table
            [linSolStr,linSolFrmt] = hisNames(this.linSol);
            % str  = {'iter', 'Jc','|x-xOld|', '|dJ|/|dJ0|','iterCG','relresCG','mu','LS'};
            str = [{'iter', 'Jc','|x-xOld|', '|dJ|/|dJ0|'},linSolStr,{'mu','LS'}];
            frmt = [{'%-12d','%-12.2e','%-12.2e','%-12.2e'},linSolFrmt,{'%-12.2e','%-12d'}];
        end
        
        function [xc,His] = solve(this,fctn,xc,fval)
            % minimizes fctn starting with xc. (optional) fval is printed
            if not(exist('fval','var')); fval = []; end;
            
            [str,frmt] = hisNames(this); % display text
            numNames = length(str);
            
            % parse objective functions, which is a dnnVarProObjFctn or a classObjFctn. The
            % goal is to call the function specified by the object fctn as
            % fctn(xc)
            % parseObjFctn just returns function handles for evaluation and
            % printing
            % The method eval is defined in the object dnnVarProObjFctn
            [fctn,objFctn,objNames,objFrmt,objHis]     = parseObjFctn(this,fctn);
            str = [str,objNames{:}]; frmt = [frmt,objFrmt{:}];
            [fval,obj2Fctn,obj2Names,obj2Frmt,obj2His] = parseObjFctn(this,fval);
            str = [str,obj2Names{:}]; frmt = [frmt,obj2Frmt{:}];
            doVal     = not(isempty(obj2Fctn));
            
            % evaluate training and validation
            % compute the gradients and the Hessian
            % Jc: cost of L (total cost in (2.4) of stable DNN paper) when W is elimited, i.e., Jc=min_W L(W, K)
            
            % dJ= gradient of Jc w.r.t the neural weights K 
            
            % d2J: hessian of Jc w.r.t the neural weights K
            [Jc,para,dJ,d2J,PC] = fctn(xc); pVal = [];
            if doVal
                if isa(objFctn,'dnnVarProBatchObjFctn') || isa(objFctn,'dnnVarProObjFctn')
                    %[Fval,pVal] = fval([xc; para.infeasi(:)]);
                    [Fval,pVal] = fval([xc; para.W(:)]);
                else
                    [Fval,pVal] = fval(xc);
                end
            end
            
            %%
            % PRINT DETAILS OF ITERATIONS
            % if the output is set to verbose print the summary of newton
            % parameters
            
            if this.out>0
                fprintf('== newton (n=%d,maxIter=%d,maxStep=%1.1e) ===\n',...
                    numel(xc), this.maxIter, this.maxStep);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
            end
            
            %% Start with the newton's iterations
            
            % his is a storeplace for the output of each iteration (fnct
            % value, magnitude of the step etc ...
            his = zeros(1,numel(str));
            
            nrm0 = norm(dJ(:));
            iter = 1;
            xOld = xc;
            while iter <= this.maxIter

                hisEnd = 4;
                his(iter,1:hisEnd)  = [iter,gather(Jc),gather(norm(xOld(:)-xc(:))),gather(norm(dJ(:))/nrm0)];
                if this.out>0
                    fprintf([frmt{1:4}], his(iter,1:4));
                end
                
                % TERMINATION condition
                if (norm(dJ(:))/nrm0 < this.rtol) || (norm(dJ(:))< this.atol), break; end
                
                % SOLVE the linear system
                % [s,~,relresCG,iterCG,resvec] = solve(this.linSol,d2J,-dJ(:),[],PC);
                % call the method solve of the object this.linSol. For
                % instance this is a steihaugPCG object
                % s is the solution
                
                % liang: use conjugate gradient method to solve linear
                % equations to obtain newton Direction, here $s$ represents
                % the newton Direction
                [s,linSolPara] = solve(this.linSol,d2J,-dJ(:),[],PC);
                
                % "his" is the variable storing data about iterations of
                % the optimization method. hisVals extract his from the
                % object linSolPara
                linSolVals = hisVals(this.linSol,linSolPara);
                
                % if s=0 exit
                if norm(s) == 0, s = -dJ(:)/norm(dJ(:)); end
                clear d2J
				clear PC

                % PRINT iteration stuff on the command window
                % his(iter,5:6) = [iterCG, relresCG];
                numPrintOuts = length(linSolVals);
                hisStart = hisEnd+1;
                hisEnd = hisEnd+numPrintOuts;
                if numPrintOuts
                    his(iter,hisStart:hisEnd) = linSolVals;
                    if this.out>0
                        fprintf([frmt{hisStart:hisEnd}], his(iter,hisStart:hisEnd));
                    end
                end
                
                % s = d2F\(-dF(:));
                if max(abs(s(:))) > this.maxStep
                    s = s/max(abs(s(:))) * this.maxStep; 
                end
                % LINE SEARCH: select the scalinh mu
                % set mu=1 in the first iteration
                if iter == 1; mu = 1.0; end
                
                % call the method lineSearch of the object this.LS
                % for instance, this can be an object of type Armijo
                
                % liang: use this step to find the optimal stepsize for
                % Newton optimization method on direction $s$
                [xt,mu,lsIter] = lineSearch(this.LS,fctn,xc,mu,s,Jc,dJ);
                if (lsIter > this.LS.maxIter)
                    disp('LSB in newton'); %keyboard
                    his = his(1:iter,:);
                    break;
                end
                
                %%% PRINT ADDITIONAL info about the current iteration on
                %%% the screen
                hisStart = hisEnd+1;
                hisEnd = hisEnd+2;
                his(iter,hisStart:hisEnd) = [mu lsIter];
                if this.out>0
                    fprintf([frmt{hisStart:hisEnd}], his(iter,hisStart:hisEnd));
                end
                if lsIter == 1
                    mu = min(mu*1.5,1);
                end
                xOld       = xc;
                xc         = xt;
                if doVal
                    if isa(objFctn,'dnnVarProBatchObjFctn') || isa(objFctn,'dnnVarProObjFctn')
                        [Fval,pVal] = fval([xc; para.W(:)]);
                    else
                        [Fval,pVal] = fval(xc);
                    end
                end
                [Jc,para,dJ,d2J,PC] = fctn(xc);
				
                if not(isempty(objNames)) || not(isempty(obj2Names)) 
                    hisStart = hisEnd+1;
                    his(iter,hisStart:end) = [gather(objHis(para)), gather(obj2His(pVal))];
                    if this.out>0
                        fprintf([frmt{hisStart:end}],his(iter,hisStart:end));
                    end
                end
                if this.out>0
                    fprintf('\n');
                end
                %%% INCREMENT the iteration counter and start a new Newton
                %%% step
                iter = iter + 1;
            end
            His = struct('str',{str},'frmt',{frmt},'his',his(1:min(iter,this.maxIter),:));
        end
    end
end