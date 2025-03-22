clear
clc
addpath(genpath(pwd))

load bbcsportIncomplete.mat
numView = length(data);
nCluster = length(unique(truelabel{1}));
n = length(truelabel{1});
omega = ones(1,numView);


k = 15;  
alpha = 10;

%% Dataset Normalization and Initialization
for i = 1:numView
    data{i} = data{i}./repmat(sqrt(sum(data{i}.^2,1)),size(data{i},1),1);  %normalized
    A{i} = zeros(n,n);
    A{i} = SimilarityGeneration(data{i}, k, 0);
end

A = SimilarityCompletionAverage(A, index, 1);

for i = 1 :numView
    NanIdx = isnan(A{i});
    A{i}(NanIdx) = 0;
end

[L,Lv,predictLabel] = FusionSum(A, nCluster, numView);

FinalResult = ClusteringMeasure(truelabel{1}, predictLabel)



maxIter = 1;  

for iter = 1:maxIter

    %% Feature learning on complete data;
    for v = 1:numView
        [U{v}] = GraphFiltering(data{v}, A{i}, index{v}, alpha);
    end

    %% Incomplete data represetation inferring
    for v = 1:numView
        Delta = diag(ones(size(Lv{v},1),1)) - Lv{v};
        UnKnownIdx{v} = setdiff([1:n], index{v});
        U{v}(:,UnKnownIdx{v}) = -U{v}(:,index{v})*Delta(index{v},UnKnownIdx{v})/Delta(UnKnownIdx{v},UnKnownIdx{v});
    end

    %U = NormalizeFeature(U,numView);
    diffusion_k = 30; 
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

% ind = 2;
% Representation = U{ind}';
% label = truelabel{ind};
% classlabel = ones(length(label),1);
% classlabel(UnKnownIdx{ind}) = 2;
% tsne(Representation, classlabel, 2)