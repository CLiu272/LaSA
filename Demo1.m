clear
clc
addpath(genpath(pwd))

load 3sourceIncomplete.mat
numView = length(data);
nCluster = length(unique(truelabel{1}));
n = length(truelabel{1});
omega = ones(1,numView);


k = 15;  % 5 for ORL;  10 for bbcsport, BUAA; 
alpha = 1;
pho = 0.1; 
beta = 1;
mu = 2;
sX = [n, n, numView]; 

%% Dataset Normalization and Initialization
for i = 1:numView
    data{i} = data{i}./repmat(sqrt(sum(data{i}.^2,1)),size(data{i},1),1);  %normalized
    A{i} = zeros(n,n);
    %GraphTmp = constructW_PKN(data{i}(:,index{i}),k);
    A{i} = SimilarityGeneration(data{i}, k, 0);
    %A{i}(index{i},index{i}) = GraphTmp;
end

A = SimilarityCompletionAverage(A, index, 1);

for i = 1 :numView
    NanIdx = isnan(A{i});
    A{i}(NanIdx) = 0;
end

[L,Lv,predictLabel] = FusionSum(A, nCluster, numView);

FinalResult = ClusteringMeasure(truelabel{1}, predictLabel)

maxIter = 5;  % 3 for Yale  Wiki
infIter = 5;

for iter = 1:maxIter

    %% Feature learning on complete data;
    for v = 1:numView
        [U{v}] = GraphFiltering(data{v}, A{i}, index{v}, alpha);
    end

    %% Incomplete data represetation inferring
    %Delta = diag(ones(size(L,1),1)) - L;

    for v = 1:numView
        Delta = diag(ones(size(Lv{v},1),1)) - Lv{v};
        UnKnownIdx{v} = setdiff([1:n], index{v});
        U{v}(:,UnKnownIdx{v}) = -U{v}(:,index{v})*Delta(index{v},UnKnownIdx{v})/Delta(UnKnownIdx{v},UnKnownIdx{v});
    end


    %     for v=1:numView
    %         Utmp = U{v};
    %         for iter = 1:infIter
    %             UnKnownIdx{v} = setdiff([1:n], index{v});
    %             U{v}(:,UnKnownIdx{v}) = 0;
    %             U{v} = U{v}*L;
    %             U{v}(:,index{v}) = Utmp(:,index{v});
    %         end
    %     end

    U = NormalizeFeature(U,numView);

%     for i = 1:numView
%         ed = L2_distance_1(U{i}, U{i});
%         temp_A=zeros(n);
%         B = J{i}-Q{i}/pho;
%         for j = 1:n
%             ad = (pho*B(j,:)-ed(j,:))/(2*alpha+pho);
%             temp_A(j,:) = EProjSimplex_new(ad);
%         end
%         A{i} = temp_A;
%         A{i} = A{i} - diag(diag(A{i}));
%         NanIdx = isnan(A{v});
%         A{i}(NanIdx) = 0;
%     end
% 
%     % == update J{i} ==
%     A_tensor = cat(3, A{:,:});
%     Q_tensor = cat(3, Q{:,:});
%     a = A_tensor(:);
%     q = Q_tensor(:);
%     [j, ~] = wshrinkObj(a+1/pho*q,beta/pho,sX,0,3,omega);
%     J_tensor = reshape(j, sX); 
%     for i=1:numView
%         J{i} = J_tensor(:,:,i);
%     end
%         
%    % == update Q{i} ==
%     for i=1:V
%         Q{i} = Q{i}+pho*(A{i}-J{i});
%     end
% 
%     % == update pho ==
%     pho = pho*mu;
    diffusion_k = 20; % 20 for MSRC bbcsports; 5 for ORL; 25 for wiki bbcsports; 10 for BUAA;
    for v = 1 :numView
        A{v} = SimilarityGeneration(U{v}, k, 0);
        A{v} = ConsistencyDiffusionTPGKNN(A{v},L,diffusion_k);
        NanIdx = isnan(A{v});
        A{v}(NanIdx) = 0;
    end



    %% Obtain unified similarity matrix by PIC approach
    [L,Lv,predictLabel] = FusionSum(A, nCluster, numView);

    FinalResult = ClusteringMeasure(truelabel{1}, predictLabel)

end

% ind = 1;
% Representation = U{ind}(:,index{ind})';
% label = truelabel{ind}(index{ind});
% tsne(Representation, label, 2)

ind = 1;
Representation = U{ind}';
%Representation = Representation(UnKnownIdx{ind},:);
label = truelabel{ind};
%label = label(UnKnownIdx{ind});
%label(UnKnownIdx{ind}) = label(UnKnownIdx{ind})+5;
tsne(Representation, label, 2)
