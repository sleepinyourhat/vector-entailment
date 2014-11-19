% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ trainDataset, testDatasetsCell, trainingLengths ] = LoadConstitDatasets (wordMap, relationMap, hyperParams)
% Load and combine all of the training and test data.
% This is slow. And can probably be easily improved if it matters.

% trainFilenames: Load these files as training data.
% testFilenames: Load these files as test data.
% splitFilenames: Split these files into train and test data.

% relationIndices: An optional matrix with three rows, one each for 
% train/test/split, indicating which set of relations the dataset uses.

PERCENT_USED_FOR_TESTING = hyperParams.testFraction;

if hyperParams.fragmentData
    trainDataset = hyperParams.trainFilenames;
else
    trainDataset = [];
end
testDatasets = {};
relationIndex = 1;

trainingLengths = [];

for i = 1:length(hyperParams.trainFilenames)
    Log(hyperParams.statlog, ['Loading training dataset ', hyperParams.trainFilenames{i}]);
    if isfield(hyperParams, 'relationIndices')
        relationIndex = hyperParams.relationIndices(1, i);
    end

    if ~hyperParams.fragmentData
        dataset = LoadConstitData(hyperParams.trainFilenames{i}, wordMap, relationMap, ...
                                  hyperParams, false, relationIndex);
        trainDataset = [trainDataset; dataset];
        trainingLengths = [trainingLengths; length(dataset)];
    else
        LoadConstitData(hyperParams.trainFilenames{i}, wordMap, relationMap, hyperParams, true, relationIndex);
    end
        
end

for i = 1:length(hyperParams.testFilenames)
    if isfield(hyperParams, 'relationIndices')
        relationIndex = hyperParams.relationIndices(2, i);
    else
        relationIndex = 1;
    end

    Log(hyperParams.statlog, ['Loading test dataset ', hyperParams.testFilenames{i}]);
    dataset = LoadConstitData(hyperParams.testFilenames{i}, wordMap, relationMap, hyperParams, false, relationIndex);
    testDatasets = [testDatasets, {dataset}];
end

for i = 1:length(hyperParams.splitFilenames)
    if isfield(hyperParams, 'relationIndices')
        relationIndex = hyperParams.relationIndices(3, i);
    else
        relationIndex = 1;
    end

    Log(hyperParams.statlog, ['Loading split dataset ', hyperParams.splitFilenames{i}]);
    dataset = LoadConstitData(hyperParams.splitFilenames{i}, wordMap, relationMap, hyperParams, false, relationIndex);
    lengthOfTestPortion = ceil(length(dataset) * PERCENT_USED_FOR_TESTING);
    startOfTestPortion = 1 + (hyperParams.foldNumber - 1) * lengthOfTestPortion;
    endOfTestPortion = min(hyperParams.foldNumber * lengthOfTestPortion, length(dataset));
    testPortion = dataset(startOfTestPortion:endOfTestPortion);
    trainPortion = [dataset(1:(startOfTestPortion - 1)); dataset(endOfTestPortion + 1:length(dataset))];
    testDatasets = [testDatasets, {testPortion}];
    
    % TODO - make fragment-safe
    trainDataset = [trainDataset; trainPortion];
    assert(length(testPortion) == lengthOfTestPortion);
    assert(length(testPortion) + length(trainPortion) == length(dataset));
end

% Evaluate on test datasets, and show set-by-set results
datasetNames = [hyperParams.testFilenames, hyperParams.splitFilenames];
testDatasetsCell = {datasetNames, testDatasets};