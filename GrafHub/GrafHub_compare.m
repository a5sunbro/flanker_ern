


function [H]=EEG_GAD(A0,f0)
%  Input: A0: Adjacency matrix NxN
%         f0: Graph signal matrix Nxp
%  Output: H: Filter response
%  Author: Meiby Ortiz-Bouza
%  Address: Michigan State University, ECE
%  email: ortizbou@msu.edu

%%% Parameters
T=3;  
rho=1;
alpha=0.5;
hubnodes_ocvsm = zeros(64,20);
hubnodes_pca = zeros(64,20);
hubnodes_lof = zeros(64,20);
hubnodes_rrcforest = zeros(64,20);
hubnodes_unfiltered = zeros(64, 20);
hubnodes_betweenness = zeros(64, 20);
hubnodes_closeness = zeros(64, 20);
hubnodes_degree = zeros(64,20);
hubnodes_eigenvector = zeros(64,20);
for patient=1:20
f = squeeze(f0(patient,:,:));
A = squeeze(A0(patient,:,:));

[~,N]=size(A);
[~,p]=size(f);  




names = [
            "FP1"
            "AF7"
            "AF3"
            "F1"
            "F3"
            "F5"
            "F7"
            "FT7"
            "FC5"
            "FC3"
            "FC1"
            "C1"
            "C3"
            "C5"
            "T7"
            "TP7"
            "CP5"
            "CP3"
            "CP1"
            "P1"
            "P3"
            "P5"
            "P7"
            "P9"
            "PO7"
            "PO3"
            "O1"
            "IZ"
            "OZ"
            "POZ"
            "PZ"
            "CPZ"
            "FPZ"
            "FP2"
            "AF8"
            "AF4"
            "AFZ"
            "FZ"
            "F2"
            "F4"
            "F6"
            "F8"
            "FT8"
            "FC6"
            "FC4"
            "FC2"
            "FCZ"
            "CZ"
            "C2"
            "C4"
            "C6"
            "T8"
            "TP8"
            "CP6"
            "CP4"
            "CP2"
            "P2"
            "P4"
            "P6"
            "P8"
            "P10"
            "PO8"
            "PO4"
            "O2"
        ];
G = graph(A, names);


nd_rank_degree = centrality(G,'degree', 'Importance', G.Edges.Weight);
[~,pred] = maxk(nd_rank_degree,2);
hubnodes_degree(pred, patient) = 1;


nd_rank_eigenvector = centrality(G,'eigenvector', 'Importance', G.Edges.Weight);
[~,pred] = maxk(nd_rank_eigenvector,2);
hubnodes_eigenvector(pred, patient) = 1;

nd_rank_betweenness = centrality(G,'betweenness', 'Cost', G.Edges.Weight);
[~,pred] = maxk(nd_rank_betweenness,2);
hubnodes_betweenness(pred, patient) = 1;


nd_rank_closeness = centrality(G,'closeness', 'Cost', G.Edges.Weight);
[~,pred] = maxk(nd_rank_closeness,2);
hubnodes_closeness(pred, patient) = 1;



%% Learn filter
An = normadj(A);   % normalized Adjacency
Ln = eye(N)-An;   % normalized Laplacian
[U,d]=eig(full(Ln));
D=diag(d);


%%%  t-th shifted input signal as S(t) := U'*D^t*U'*F
for t=1:T
zt{t}=U*d^(t-1)*U'*f;
end

for i=1:N
    for t=1:T
    zn(t,:,i)=zt{t}(i,:);
    end
end

ZN1 = permute(sum(pagemtimes(permute(zn, [2 3 1]), pagemtimes(Ln,permute(zn, [3 2 1]))), [1 2]), [3 1 2]);

%% Initializations
mu1=rand(N,p);
V=mu1/rho;
h=rand(T,1);
h=h/norm(h);
H=0;
for t=1:T
    Hnew=H+h(t)*diag(D.^(t-1));
    H=Hnew;  
end

thr=alpha/rho;
for n=1:40
    %% ADMM (Z,h,V)
    %%% B^(k+1) update using h^k and V^k
    X=(eye(N)-U*H*U')*f-V;
    B=wthresh(X,'s',thr);
    %%% h^(k+1) update using B^(k+1) and V^k
    E=B-f+V;
    count1=0;
    count2=0;
    SZ=0;
    for t=1:p
    for k=1:N
        count1=0;
        SZnew=SZ+sum(ZN1,3);
        SZ=SZnew;
        count2=count2+1;
        ZN2(:,:,count2)=zn(:,t,k)*zn(:,t,k)';
        b(:,:,count2)=zn(:,t,k)*E(k,t);
    end
    end
    Y=2*SZ+rho*sum(ZN2,3);
    h_new=-inv(Y)*rho*sum(b,3);
    h_new=h_new/norm(h_new);

    H=0; %% C filter for next iteration
    for t=1:T
        Hnew=H+h_new(t)*diag(D.^(t-1));
        H=Hnew;  
    end

    %%% V^(k+1) update using V^k, Z^(k+1), and c^(k+1)
    V_new=V+rho*(B-(eye(N)-U*H*U')*f);
    if norm(h_new-h)<10^-3
        break
    end
    h=h_new;
    V=V_new;
end
clear b ZN2


f_tilde=U*H*U'*f;




scores=vecnorm(f');
scores = zscore(scores, 1, 'all');
threshold = 3;
pred = find(scores > threshold| scores < -threshold);
hubnodes_unfiltered(pred,patient) = 1;
clear e0
clear en
clear s



% 
contam_frac = 0.03125;
threshold = 3;
supmachine = ocsvm(f,ContaminationFraction=contam_frac);
tf = isanomaly(supmachine, f);
% disp("OCVSM:");
% disp(names(find(tf == 1)));
hubnodes_ocvsm(find(tf==1),patient) = 1;
warning('off', 'all'); %to turn off PCA warning

[coeff,score,pcvar,mu,v,S] = pca(f);
t = score*coeff' + repmat(mu, [1 50]);   %t is the reconstructed signal from the principle components
scores = vecnorm(t' - f');
scores = zscore(scores, 1, 'all');
% 
% disp("PCA:");
% disp(names(find(scores > threshold | scores < -threshold)));
hubnodes_pca(find(scores > threshold | scores < -threshold),patient) = 1;

% forest = iforest(f, ContaminationFraction=0.05);
% tf = isanomaly(forest, f);
% disp(names(find(tf == 1)));
% 
lof_obj = lof(f,ContaminationFraction=contam_frac);
tf = isanomaly(lof_obj,f);
% disp("LOF:");
% disp(names(find(tf == 1)));
hubnodes_lof(find(tf==1),patient) = 1;


forest = rrcforest(f, ContaminationFraction=contam_frac);
tf = isanomaly(forest,f);
% disp("RRCforest:");
% disp(names(find(tf == 1)));
hubnodes_rrcforest(find(tf==1),patient) = 1;



end
save("results_eigenvector.mat", 'hubnodes_eigenvector');
save('results_degree.mat', "hubnodes_degree");
save("results_closeness.mat", 'hubnodes_closeness');
save('results_betweenness.mat', "hubnodes_betweenness");
save("results_unfiltered.mat", 'hubnodes_unfiltered');
save('results_pca.mat', "hubnodes_pca");
save('results_lof.mat', "hubnodes_lof");
save('results_rrcforest.mat', "hubnodes_rrcforest");
save('results_ocvsm.mat', "hubnodes_ocvsm");
end
