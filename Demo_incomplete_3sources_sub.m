clear
clc
addpath(genpath(pwd))

load 3sources_Per0.2.mat
numView = length(data);
nCluster = length(unique(truelabel{1}));
n = length(truelabel{1});
omega = ones(1,numView);


k = 15;  % 5 for ORL;  10 for bbcsport, BUAA; 
alpha = 5;


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

    %U = NormalizeFeature(U,numView);

    diffusion_k = 50; % 20 for MSRC bbcsports; 5 for ORL; 25 for wiki bbcsports; 10 for BUAA;
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


