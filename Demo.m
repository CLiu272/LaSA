clear
clc
addpath(genpath(pwd))

load 3sourceIncomplete.mat
numView = length(data);
nCluster = length(unique(truelabel{1}));
n = length(truelabel{1});

k = 10;
alpha = 0.1;

%% Dataset Normalization
data = NormalizeFeature(data,numView);

%% Initialization (individual similarity matrix, unified similarity matrix and vector V)
[L,V,Q,predictLabel] = Initialization(data,index,nCluster,k);

FinalResult = ClusteringMeasure(truelabel{1}, predictLabel)

for iter = 1:10

    %% Feature learning on complete data;
    for v = 1:numView
        [U{v}] = GraphFiltering(data{v}, L , index{v}, alpha);
    end
    %% Incomplete data represetation inferring
    Delta = diag(ones(size(L,1),1)) - L;
    for v = 1:numView
        UnKnownIdx{v} = setdiff([1:n], index{v});
        U{v}(:,UnKnownIdx{v}) = -U{v}(:,index{v})*Delta(index{v},UnKnownIdx{v})/Delta(UnKnownIdx{v},UnKnownIdx{v});
    end
    
    %U = NormalizeFeature(U,numView);
    for i = 1 :numView
        Q{i} = SimilarityGeneration(U{i}, k, 0);
    end

    Q = SimilarityCompletionAverage(Q, index, 1);

    for i = 1 :numView
        NanIdx = isnan(Q{i});
        Q{i}(NanIdx) = 0;
    end


    %% Obtain unified similarity matrix by PIC approach
    [L,V,predictLabel] = FusionSum(Q, nCluster, numView);
    FinalResult = ClusteringMeasure(truelabel{1}, predictLabel)

end

