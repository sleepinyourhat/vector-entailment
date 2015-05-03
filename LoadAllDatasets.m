% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ trainDataset, testDatasetsCell, trainingLengths ] = LoadAllDatasets(wordMap, labelMap, hyperParams)
% Load and combine all of the training and test data.
% This is slow. And can probably be easily improved if it matters.

% trainFilenames: Load these files as training data.
% testFilenames: Load these files as test data.
% splitFilenames: Split these files into train and test data.

% labelIndices: An optional matrix with three rows, one each for 
% train/test/split, indicating which set of labels the dataset uses.

if hyperParams.SSTMode
    loadFileFn = @LoadSSTData;
elseif hyperParams.sentenceClassificationMode
    loadFileFn = @LoadSentenceClassificationData;
else
    loadFileFn = @LoadEntailmentData;
end

if hyperParams.fragmentData
    trainDataset = hyperParams.trainFilenames;
else
    trainDataset = [];
end
testDatasets = {};
labelIndex = 1;

trainingLengths = [];

for i = 1:length(hyperParams.trainFilenames)
    Log(hyperParams.statlog, ['Loading training dataset ', hyperParams.trainFilenames{i}]);
    if isfield(hyperParams, 'labelIndices')
        labelIndex = hyperParams.labelIndices(1, i);
    end

    if ~hyperParams.fragmentData
        dataset = loadFileFn(hyperParams.trainFilenames{i}, wordMap, labelMap, ...
                                  hyperParams, false, labelIndex);
        trainDataset = [trainDataset; dataset];
        trainingLengths = [trainingLengths; length(dataset)];
    else
        loadFileFn(hyperParams.trainFilenames{i}, wordMap, labelMap, hyperParams, true, labelIndex);
    end
        
end

for i = 1:length(hyperParams.testFilenames)
    if isfield(hyperParams, 'labelIndices')
        labelIndex = hyperParams.labelIndices(2, i);
    else
        labelIndex = 1;
    end

    Log(hyperParams.statlog, ['Loading test dataset ', hyperParams.testFilenames{i}]);
    dataset = loadFileFn(hyperParams.testFilenames{i}, wordMap, labelMap, hyperParams, false, labelIndex);
    testDatasets = [testDatasets, {dataset}];
end

for i = 1:length(hyperParams.splitFilenames)
    if isfield(hyperParams, 'labelIndices')
        labelIndex = hyperParams.labelIndices(3, i);
    else
        labelIndex = 1;
    end

    Log(hyperParams.statlog, ['Loading split dataset ', hyperParams.splitFilenames{i}]);
    dataset = loadFileFn(hyperParams.splitFilenames{i}, wordMap, labelMap, hyperParams, false, labelIndex);

    lengthOfTestPortion = ceil(length(dataset) * hyperParams.testFraction);
    startOfTestPortion = 1 + (hyperParams.foldNumber - 1) * lengthOfTestPortion;
    endOfTestPortion = min(hyperParams.foldNumber * lengthOfTestPortion, length(dataset));
    
    testPortion = dataset(startOfTestPortion:endOfTestPortion);
    testDatasets = [testDatasets, {testPortion}];
    
    if ~(isfield(hyperParams, 'specialAndOrMode') && i > ((2 * hyperParams.specialAndOrMode) + 1))
        firstTrainPortion = dataset(1:(startOfTestPortion - 1));
        secondTrainPortion = dataset(endOfTestPortion + 1:length(dataset));
        trainPortion = [firstTrainPortion; secondTrainPortion];
        trainDataset = [trainDataset; trainPortion];
        trainingLengths = [trainingLengths; length(dataset)];
    else
        Log(hyperParams.statlog, ['Discarding train portion of split dataset ', hyperParams.splitFilenames{i}]);
    end
end

datasetNames = [hyperParams.testFilenames, hyperParams.splitFilenames];
testDatasetsCell = {datasetNames, testDatasets};
