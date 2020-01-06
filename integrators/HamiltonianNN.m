classdef HamiltonianNN < abstractMeganetElement
    % Double Layer Hamiltonian block
    %
    % Z_k+1 = Z_k - h*act(K(theta_k)'*Y_k + b),  
    % Y_k+1 = Y_k + h*act(K(theta_k)* Z_k+1 + b) 
    %
    % theta_k is the sampling time t_k
    %
    % The input features are divided into Y and Z here based on the sizes 
    % of K.
    %
    % References:
    %
    % Chang B, Meng L, Haber E, Ruthotto L, Begert D, Holtham E: 
    %      Reversible Architectures for Arbitrarily Deep Residual Neural Networks, 
    %      AAAI Conference on Artificial Intelligence 2018
    
    properties
        activation
        K
        B
        nt
        h
        useGPU
        precision
    end
    
    methods
        function this = HamiltonianNN(activation,K,B,nt,h,varargin)
            % Object creator
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU    = [];
            precision = [];
            % varargin are optional parameters thaht can be used for overriding the default values.
            % Example: varargin={'precision',0.001) sets precision=0.001 instead of the default precision=[] 
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.activation = activation;
            this.K = K;
            this.B = B;
            this.nt       = nt;
            this.h        = h;
        end
        
        function n = nTheta(this)
            n = this.nt*(nTheta(this.K)+ sizeLastDim(this.B));
        end
        
        function n = sizeFeatIn(this)
            n = sizeFeatOut(this.K);
%             n2 = sizeFeatIn(this.K);
%             if (numel(n1) > 2) && (numel(n2) > 2)
%                 % convolution layer. add chanels together
%                 n = n1;
%                 n(3) = n1(3) + n2(3);
%             else
%                 n = n1+n2;            
%             end
        end
        function n = sizeFeatOut(this)
            n = sizeFeatIn(this.K);
        end

        
        function theta = initTheta(this)
            theta = repmat([vec(initTheta(this.K));...
                            zeros(sizeLastDim(this.B),1)],this.nt,1);
        end
        
        function [net2,theta2] = prolongateWeights(this,theta)
            % piecewise linear interpolation of network weights 
            t1 = 0:this.h:(this.nt-1)*this.h;
            
            net2 = HamiltonianNN(this.activation,this.K,this.B,2*this.nt,this.h/2,'useGPU',this.useGPU,'precision',this.precision);
          
            t2 = 0:net2.h:(net2.nt-1)*net2.h;
            
            theta2 = inter1D(theta,t1,t2);
        end
        
        function [thetaK,thetaB] = split(this,x)
            % x stores, in a vector form [vec(K_1);vec(b_1),...,
            % vec(K_nt);vec(b_nt)], i.e. weights and biases of all layers
           x   = reshape(x,[],this.nt);
           % after reshaping, x is a matrix with each column i  equal to [vec(K_i);vec(b_i)]
           thetaK = x(1:nTheta(this.K),:);
           % thetaK is a matrix with each column i  equal to vec(K_i)
           thetaB = x(nTheta(this.K)+1:end,:);
           % thetaB is a matrix with each column i  equal to vec(B_i)
        end

               % ------- forwardProp forward problems -----------
        function [Y,tmp] = forwardProp(this,theta,Y0,varargin)
            % forward propagation of the set of input data stored as columns of the matrix Y0
            % theta contains in a vectorized form the parameters of all
            % layers, i.e. theta=[vec(K_1);vec(b_1),...,
            % vec(K_nt);vec(b_nt)]
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            % set the initial state of the DT evolution of the hamiltonian
            % system as
            % Y=copy of the input data
            % Z=0
            Y = Y0;
            Z = 0*Y;
            [thetaK,thetaB] = split(this,theta);
            % for each layer, extract K and B and compute Z^+ and Y^+ 
            % previous states are overwritten and only the final state
            % at time nt+1 is stored in Y and Z
            for i=1:this.nt
                % extract the matrix K from the vectorized form thetaK
                Ki = getOp(this.K,thetaK(:,i)); 
                % extract the bias b from the vectorized form thetaB
                bi = this.B*thetaB(:,i);
                
                fY = this.activation(Ki'*Y + bi);
                Z  = Z - this.h*fY;
                
                fZ = this.activation(Ki*Z + bi);
                Y  = Y + this.h*fZ;
            end
            % tmp stores {Output Y of last layer, Output Z of last layer,
            % input Y0 to the DNN}
            tmp = {Y,Z,Y0};
        end
        
        % -------- Jacobian matvecs ---------------
%         function dX = JYmv(this,dX,theta,~,tmp)
%             if isempty(dX) || (numel(dX)==0 && dX==0.0)
%                 dX     = zeros(size(Y),'like',Y);
%                 return
%             end
%             X0 = tmp{3};
%             [Y,Z] = splitData(this,X0);
%             [dY,dZ]   = splitData(this,dX);
%             [th1,th2] = split(this,theta);
%             for i=1:this.nt
%                 [fZ,tmp] = forwardProp(this.layer1,th1(:,i),Z,'storeInterm',1);
%                 dY = dY + this.h*JYmv(this.layer1,dZ,th1(:,i),Z,tmp);
%                 Y  = Y + this.h*fZ;
%                 
%                 [fY,tmp] = forwardProp(this.layer2,th2(:,i),Y);
%                 dZ = dZ - this.h*JYmv(this.layer2,dY,th2(:,i),Y,tmp);
%                 Z = Z - this.h*fY;
%             end
%             dX = unsplitData(this,dY,dZ);
%         end
        
        function dY = Jmv(this,dtheta,dY,theta,~,tmp)
            % Given a minibatch Y, net parameters theta and the increments
            % dY dtheta, compute the increment dY
            
            % All comments hereafter assume that Y is a single datapoint
            % (column vector)
            
            % define the initial conditions dy_{0} and y_{0} 

            if isempty(dY)
                dY = 0*tmp{1};
            end
            
            % Y stores the input data to the first layer
            Y = tmp{3};
            
            % define the initial conditions z_{-1/2} and dz_{-1/2} 
            Z = 0*Y;
            dZ = 0*Z;
            
            [thK,thB]   = split(this,theta);
            [dthK,dthB] = split(this,dtheta);
            for i=1:this.nt
                % extract matrices Ki,dKi and biases bi and dbi
       
                 Ki  = getOp(this.K,thK(:,i));
                 % dKi is a vector storing the increment in Ki
                 dKi = getOp(this.K,dthK(:,i));
                 bi  = this.B*thB(:,i);
                 % dbi is a vector storing the increment in b_i 
                
                 dbi = this.B*dthB(:,i);
                 
                 % dfY is a vector storing the induced increment 
                 %  d sigma(K_i'* y_i+b_i)
                 % NOTE: this increment is related to the update of the
                 % z-variable in Ruthotto's paper
                 [fY,dfY]  = this.activation(Ki'*Y + bi);
       
                 % JY is a vector storing the induced increment
                 %       d \sigma*(d K_i'*y_i+K_i'*d y_i+d b_i)
                 % NOTE: this increment is related to the update of the
                 % z-variable in Ruthotto's paper

                 JY = dfY.*(dKi'*Y+Ki'*dY+dbi);
                 
                 % dZ is a vector storing the increment 
                 %       d z_{i+1/2}= d z_{i-1/2} - h*d \sigma
                 dZ = dZ - this.h*JY;
                 
                 % Z is a vector storing the update
                 %       z_{i+1/2}= z_{i-1/2} - h*\sigma
                
                 Z  = Z  - this.h*fY;
                 
                 % fZ =sigma(K_i z_{i+1/2} +b_i)
                 
                 % dfZ is a vector storing the increment
                 %  d sigma(K_i z_{i+1/2} +b_i)
                 
                 [fZ,dfZ] = this.activation(Ki*Z + bi);
                 
                 % JZ is a vector storing the induced increment
                 % d \sigma*(d K_i*z_{i+1/2}+K_i*z_{i+1/2}+d b_i)
                 % NOTE: this increment is related to the update of the
                 % z-variable in Ruthotto's paper

                 
                 JZ = dfZ.*(dKi*Z+Ki*dZ+dbi);
                 
                 % dY is a vector storing the increment
                 %  d y_{i+1}
                 
                 dY = dY + this.h*JZ;
                 
                 % Y is a vector storing 
                 %       y_{i+1}= y_i + h*\sigma
                 Y = Y + this.h*fZ;
            end
        end
        
        % -------- Jacobian' matvecs ----------------
%         function W = JYTmv(this,W,theta,X0,tmp)
%             % nex = sizeLastDim(Y);
%             if isempty(W)
%                 WY = 0;
%                 WZ = 0;
%             elseif not(isscalar(W))
%                 [WY,WZ] = splitData(this,W);
%             end
%             
%             Y = tmp{1};
%             Z = tmp{2};
%             [th1,th2]  = split(this,theta);
%             
%             for i=this.nt:-1:1
%                 [fY,tmp] = forwardProp(this.layer2,th2(:,i),Y);
%                 dWY = JYTmv(this.layer2,WZ,th2(:,i),Y,tmp);
%                 WY  = WY - this.h*dWY;
%                 Z = Z + this.h*fY;
%                 
%                 [fZ,tmp] = forwardProp(this.layer1,th1(:,i),Z,'storeInterm',1);
%                 dWZ = JYTmv(this.layer1,WY,th1(:,i),Z,tmp);
%                 WY  = WZ + this.h*dWZ;
%                 Y = Y - this.h*fZ;
%             end
%             W = unsplitData(this,WY,WZ);
%         end
        
        function [dtheta,W] = JTmv(this,W,theta,X0,tmp,doDerivative)
            
            % the cell array tmp is assumed to be computed by forwardProp
            % and to store {Output Y of last layer, Output Z of last layer,
            % input Y0 to the first layer}


            % GOAL: evaluate the gradient of the cost w.r.t all internal
            % weights (i.e. \theta which comprises K_0,b_0,...K_nt,b_nt but
            % NOT the clasification weights W and mu, which are given and stored in W for the current value of \theta)
            % through the adjoint method (a smart way to implement
            % backpropagation)
            
            % The gradients w.r.t W and mu are skipped because the
            % optimization code is based on "variable projection" meaning
            % that first (W,mu) is optimized keeping theta fixed and then theta is optimized keeping (W,mu) fixed.
            % Here we care about the gradients needed in this second step
           
            % The gradient of the cost w.r.t (W,mu) is computed by the
            % class generating the cost - not in this class!
            
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            
            % At this point of the code 
            % W stores dYF = dF /dY(N) 
            %JUST F AND NOT THE WHOLE COST F+R
            % where 
            % F is 1/s*S(h(W*Y+\mu),C) that is the first part of the cost
            % in (2.3) in ruthotto's paper
            
            % W is also the multiplier \lambda_N, see my formula (adj 1) 
            % in the supporting sheets
            
            % Initialize, WY=W and WZ=0. These matrices should be related to
            % the components 
            % Y and Z of the state of
            % the hamiltonian system ??
            
            % init W with zeros if not passed as input argument
            if isempty(W) || numel(W)==1
                WY = 0;
                WZ = 0;
            else
                % WY stores dYF = dF /dY(N) where Y(N) is the output of the
                % last layer
                % From the adjoint method
                % WY is also the value of the multiplier lam_Y(N)
                WY = W;
                % WZ is the value of the multiplier lam_Z(N)=dF /dZ(N) = 0
                % because the cost does not depend on Z(N)
                WZ = 0*WY;
            end
            
            % Y stores Y(N), the Y-output of the last layer of the network
           
            Y = tmp{1};
            nd = ndims(Y);
            
            % Z stores the Z(n), the Z-output of the last layer of the network
            Z = tmp{2};
            
            % thK is a matrix with each column i  equal to vec(K_i)
            % same for thB
            [thK,thB]   = split(this,theta);
            
            % initialize increments with zeros
            [dthK,dthB] = split(this,0*theta);
            
            % MAIN LOOP FROM THE LAST LAYER TO THE FIRST ONE
            
            for i=this.nt:-1:1
                
                % extract Ki and bi
                Ki = getOp(this.K,thK(:,i)); 
                bi = this.B*thB(:,i);
                
                % compute quantities related to the computation of Y(i+1)
                % dfZ=d\sigma(Ki*Z+bi)\ dZ 
                
                [fZ,dfZ] = this.activation(Ki*Z + bi);
                
                % COMPUTE dWZ associated to the update of WZ
                % dWZ=Ki'*(d\sigma(Ki*Z+bi)\ dZ .* lam_Y(i+1)
                % The explanation is in (adj 4) in my notes
                dWZ = Ki'*(dfZ.*WY); 
                
                % At this stage dthK(:,i) and dthB(:,i) are zeros
                % At the end of the loop for they should store dJ/d vec(K_i) and dJ/d b_i
                
                % JthetaTmv is a method of [dense] that computes dfZ.*WY*Z'
                % overall dthK(:,i)+ h*dfZ.*WY*Z'
                % = h * diag(sigma'(Ki*Z+bi))*Ki*(dfZ*WY)*Z'
                % COMPUTE THE ASSOCIATED UPDATE OF d J/d u_j - TO BE
                % UNDERSTOOD using (adj3 in my notes)
                dthK(:,i) = dthK(:,i)+ this.h*JthetaTmv(this.K,dfZ.*WY,[],Z);
                dthB(:,i) = dthB(:,i) + this.h*vec(sum(this.B'*(dfZ.*WY),nd));
                
                % update WZ so as to store \lam_W(i)
                WZ = WZ + this.h*dWZ;
                
                % compute Y(i) from Y(i+1) and fZ
                % see my first notes on HamiltonianNN
                Y  = Y - this.h*fZ;
                                
                % COMPUTE dWY associated to the update of WY
                % dfY=d\sigma(Ki'*Y+bi)\ dY 
                % The explanation is in (adj 4) in my notes

                [fY,dfY] = this.activation(Ki'*Y + bi);
                
                % COMPUTE dWY associated to the update of WY
                % The explanation is in (adj 4) in my notes
                dWY = Ki*(dfY.*WZ); 
                
                % JthetaTmv is a method of [dense] that computes dfY.*WZ*Y'
                % overall dthK(:,i)+ h*dfZ.*WY*Z'
                % = h * diag(sigma'(Ki*Z+bi))*Ki*(dfZ*WY)*Z'
                % COMPUTE THE ASSOCIATED UPDATE OF d J/d u_j - TO BE
                % UNDERSTOOD using (adj3 in my notes)
                
                dJK = reshape(JthetaTmv(this.K,dfY.*WZ,[],Y),size(Ki));
                dJK = vec(dJK');
                dthK(:,i) = dthK(:,i) - this.h*dJK;
                dthB(:,i) = dthB(:,i) - vec(sum(this.h*this.B'*(dfY.*WZ),nd));
                
                % update W so as to store \lam_W(i)
                WY = WY - this.h*dWY;
                Z  = Z + this.h*fY;
            end
            dtheta = vec([dthK;dthB]);
            
%             W = unsplitData(this,WY,WZ);
            W = WY;
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta; W(:)];
            end

        end
        function [thFine] = prolongateConvStencils(this,theta,getRP)
            % prolongate convolution stencils, doubling image resolution
            %
            % Inputs:
            %
            %   theta - weights
            %   getRP - function for computing restriction operator, R, and
            %           prolongation operator, P. Default @avgRestrictionGalerkin
            %
            % Output
            %  
            %   thFine - prolongated stencils
            
            if not(exist('getRP','var')) || isempty(getRP)
                getRP = @avgRestrictionGalerkin;
            end
            
            [th1Fine,th2Fine] = split(this,theta);
            for k=1:this.nt
                th1Fine(:,k) = vec(prolongateConvStencils(this.layer1,th1Fine(:,k),getRP));
                th2Fine(:,k) = vec(prolongateConvStencils(this.layer2,th2Fine(:,k),getRP));
            end
            thFine = vec([th1Fine;th2Fine]);
        end
        function [thCoarse] = restrictConvStencils(this,theta,getRP)
            % restrict convolution stencils, dividing image resolution by two
            %
            % Inputs:
            %
            %   theta - weights
            %   getRP - function for computing restriction operator, R, and
            %           prolongation operator, P. Default @avgRestrictionGalerkin
            %
            % Output
            %  
            %   thCoarse - restricted stencils
            
            if not(exist('getRP','var')) || isempty(getRP)
                getRP = @avgRestrictionGalerkin;
            end
            
            
           [th1Coarse,th2Coarse] = split(this,theta);
            for k=1:this.nt
                th1Coarse(:,k) = vec(restrictConvStencils(this.layer1,th1Coarse(:,k),getRP));
                th2Coarse(:,k) = vec(restrictConvStencils(this.layer2,th2Coarse(:,k),getRP));
            end
            thCoarse = vec([th1Coarse;th2Coarse]);
        end
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.K.useGPU  = value;
%                 this.K.useGPU  = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.K.precision = value;
%                 this.K.precision = value;
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.K.useGPU;
        end
        function precision = get.precision(this)
            precision = this.K.precision;
        end
        
        function runMinimalExample(~)
            act = @tanhActivation;
            K    = dense([2,2]);
            B     = ones(2,1);
            % HamiltonianNN(activation,K,B,nt,h,varargin)
            net   = HamiltonianNN(act,K,B,10,.1);
            Y = [1;1];
            % vectorized form of the Ki and bi in one layer. Note that bi=0
            theta =  [vec([2 1;1 2]);0];
            % theta format for all layers
            theta = vec(repmat(theta,1,net.nt));

            
            % Test the forward propagation of a single datapoint
            [YN,tmp] = forwardProp(net,theta,Y); 
            % I think Ys is a copy of YN but extracted from the cell array tmp
            % instead of created as in the previous line
            Ys = reshape(cell2mat(tmp(:,1)),2,[]);
            
            % TEST OPERATIONS ON A MINIBATCH
            % nex = # datapoints in the minibatch
            nex = 100;
            
            % nTheta(net) is it the size of theta (the vectorized weights
            % of all layers). 
            % mb is a random initialization of theta used also as
            % initialization for the vector storing the increment D theta
            % i.e. each column contains [D theta_i]
            
            mb  = randn(nTheta(net),1);
            
            % numelFeatIn(net) is the number of features (the dim of the
            % input space)
            
            % create a minibatch with random data
            Y0  = randn(numelFeatIn(net),nex);
            
            % FORWARD propagation using the random weights and minibatch
            [Y,tmp]   = net.forwardProp(mb,Y0);
            
            % COMPUTATION OF THE VECTORIZED increments
            
            % dmb stores in column i the increment in theta_i 
            % [d vec(K_i); d vec(b_i)] 
            % dmb is randomly generated
            dmb = reshape(randn(size(mb)),[],net.nt);
            % dY0 stores in column i the perturbation of Y0_i  
            % randomly generated
            dY0  = randn(size(Y0));
            
            % deduce the increment dY on the whole eminibatch, given dY0 and dtheta 
            % the vectorized form of dmb is passed to Jmv
            % note that the def of Jmv is this.Jmv(dtheta,dY,theta,~,tmp)
            % so Y0 in the next line is ignored. It is instead taken form
            % temp{3}
            dY = net.Jmv(dmb(:),dY0,mb,Y0,tmp);
            
            % test: for smaller and smaller increments (here controlled by
            % hh) I expect smaller and smaller differences Yt(:)-Y(:) 
            for k=1:20
                hh = 2^(-k);
                
                Yt = net.forwardProp(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Y(:));
                E1 = norm(Yt(:)-Y(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Y));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,mb,Y0,tmp);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2)/abs(t1));
        end
    end
    
end

