% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ trainDataset, testDatasetsCell ] = LoadConstitDatasets ...
    (trainFilenames, splitFilenames, testFilenames, wordMap, relationMap)
% Load and combine all of the training and test data.
% This is slow. And can probably be easily improved if it matters.

% trainFilenames: Load these files as training data.
% testFilenames: Load these files as test data.
% splitFilenames: Split these files into train and test data.

PERCENT_USED_FOR_TRAINING = 0.85;

trainDataset = [];
testDatasets = {};

for i = 1:length(trainFilenames)
    disp(['Loading training dataset ', trainFilenames{i}])
    dataset = LoadConstitData(trainFilenames{i}, wordMap, relationMap);
    trainDataset = [trainDataset; dataset];
end

for i = 1:length(testFilenames)
    disp(['Loading test dataset ', testFilenames{i}])
    dataset = LoadConstitData(testFilenames{i}, wordMap, relationMap);
    testDatasets = [testDatasets, {dataset}];
end

for i = 1:length(splitFilenames)
    disp(['Loading split dataset ', splitFilenames{i}])
    dataset = LoadConstitData(splitFilenames{i}, wordMap, relationMap);
    randomOrder = randperm(length(dataset));
    endOfTrainPortion = ceil(length(dataset) * PERCENT_USED_FOR_TRAINING);
    testDatasets = [testDatasets, ...
                    {dataset(randomOrder(endOfTrainPortion + 1:length(dataset)))}];
    trainDataset = [trainDataset; dataset(randomOrder(1:endOfTrainPortion))];
end

% Evaluate on test datasets, and show set-by-set results
datasetNames = [testFilenames, splitFilenames];
testDatasetsCell = {datasetNames, testDatasets};