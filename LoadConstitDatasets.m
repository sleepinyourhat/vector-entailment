% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ trainDataset, testDatasetsCell ] = LoadConstitDatasets ...
    (trainFilenames, splitFilenames, testFilenames, wordMap, relationMap, hyperParams)
% Load and combine all of the training and test data.
% This is slow. And can probably be easily improved if it matters.

% trainFilenames: Load these files as training data.
% testFilenames: Load these files as test data.
% splitFilenames: Split these files into train and test data.

% relationIndices: An optional matrix with three rows, one each for 
% train/test/split, indicating which set of relations the dataset uses.

PERCENT_USED_FOR_TRAINING = 0.85;

if hyperParams.fragmentData
    trainDataset = trainFilenames;
else
    trainDataset = [];
end
testDatasets = {};

for i = 1:length(trainFilenames)
    Log(hyperParams.statlog, ['Loading training dataset ', trainFilenames{i}]);
    if isfield(hyperParams, 'relationIndices')
        relationIndex = hyperParams.relationIndices(1, i);
    else
        relationIndex = 1;
    end
        
    if ~hyperParams.fragmentData
        dataset = LoadConstitData(trainFilenames{i}, wordMap, relationMap, hyperParams, false, relationIndex);
        trainDataset = [trainDataset; dataset];
    else
        LoadConstitData(trainFilenames{i}, wordMap, relationMap, hyperParams, true, relationIndex);
    end
        
end

for i = 1:length(testFilenames)
    if isfield(hyperParams, 'relationIndices')
        relationIndex = hyperParams.relationIndices(2, i);
    else
        relationIndex = 1
    end

    Log(hyperParams.statlog, ['Loading test dataset ', testFilenames{i}]);
    dataset = LoadConstitData(testFilenames{i}, wordMap, relationMap, hyperParams, false, relationIndex);
    testDatasets = [testDatasets, {dataset}];
end

for i = 1:length(splitFilenames)
    if isfield(hyperParams, 'relationIndices')
        relationIndex = hyperParams.relationIndices(3, i);
    else
        relationIndex = 1
    end

    Log(hyperParams.statlog, ['Loading split dataset ', splitFilenames{i}]);
    dataset = LoadConstitData(splitFilenames{i}, wordMap, relationMap, hyperParams, false, relationIndex);
    endOfTrainPortion = ceil(length(dataset) * PERCENT_USED_FOR_TRAINING);
    testDatasets = [testDatasets, ...
                    {dataset(endOfTrainPortion + 1:length(dataset))}];
    % TODO - make fragment-safe
    trainDataset = [trainDataset; dataset(1:endOfTrainPortion)];
end

% Evaluate on test datasets, and show set-by-set results
datasetNames = [testFilenames, splitFilenames];
testDatasetsCell = {datasetNames, testDatasets};