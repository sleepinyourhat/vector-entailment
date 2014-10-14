function [ modelState ] = TrainOnFragmentedData(CostGradFunc, trainingData, testDatasets, modelState, hyperParams, options)

% Preloaded MAT files for each training dataset, 
% outer index = sourceFilenameIndex, inner index reflect
% an arbitrary order within the framents of the source file.
openFragments = {};
fragmentFiles = {};
fragmentOrders = {};
fragmentOrderIndices = [];
sourceSizes = [];
openFragmentExampleOrders = {};
openFragmentExampleOrderIndices = [];
totalNumTrainingExamples = 0;

for sourceFilenameIndex = 1:length(trainingData)
    [pathname, filename, ext] = fileparts(trainingData{sourceFilenameIndex});
    listing = dir([pathname, '/pp-', filename, ext, '-', hyperParams.vocabName, '*.mat']);
    finalListing = dir([pathname, '/pp-', filename, ext, '-final-', hyperParams.vocabName, '*.mat']);
    assert(length(finalListing) == 1, ['Error in loading preprocessed training data ', trainingData{sourceFilenameIndex}]);
    fragmentFiles{sourceFilenameIndex} = [{listing.name}, {finalListing.name}];

    % Get the total example count for the source
    [startIndex, endIndex] = regexpi(finalListing(1).name,'[0-9]+.mat');
    sourceSizes(sourceFilenameIndex) = str2num(finalListing(1).name(startIndex(1):(endIndex(1)-4)));
    totalNumTrainingExamples = totalNumTrainingExamples + sourceSizes(sourceFilenameIndex);

    % Permute the fragments of the source
    fragmentOrders{sourceFilenameIndex} = randperm(length(fragmentFiles{sourceFilenameIndex}));
    fragmentOrderIndices(sourceFilenameIndex) = 1;

    % Open a first file and permute its examples
    fragmentIndex = fragmentOrders{sourceFilenameIndex}(fragmentOrderIndices(sourceFilenameIndex));
    filename = fragmentFiles{sourceFilenameIndex}{fragmentIndex};
    filepath = fileparts(trainingData{sourceFilenameIndex});
    disp(['Loading ' filename]);
    openFragments{sourceFilenameIndex} = load([filepath, '/', filename],'-mat');
    openFragmentExampleOrders{sourceFilenameIndex} = randperm(length(openFragments{sourceFilenameIndex}.data));
    openFragmentExampleOrderIndices(sourceFilenameIndex) = 1;
    clearvars a
end

while true
    sourceFilenameIndex = chooseDataset(trainingData, fragmentOrderIndices, fragmentOrders, sourceSizes);
    if sourceFilenameIndex == 0
        return
    end

    % Load a new file if needed
    if openFragmentExampleOrderIndices(sourceFilenameIndex) >= length(openFragments{sourceFilenameIndex}.data)
        fragmentOrderIndices(sourceFilenameIndex) = fragmentOrderIndices(sourceFilenameIndex) + 1;
        
        % Check that we haven't just exhausted this source
        if fragmentOrderIndices(sourceFilenameIndex) > length(fragmentOrders{sourceFilenameIndex})
            continue
        else
            % Open a file and permute its examples
            nextFileIndex = fragmentOrders{sourceFilenameIndex}(fragmentOrderIndices(sourceFilenameIndex));
            filename = fragmentFiles{sourceFilenameIndex}{nextFileIndex};
            filepath = fileparts(trainingData{sourceFilenameIndex});
            openFragments{sourceFilenameIndex} = load([filepath, '/', filename],'-mat');
            openFragmentExampleOrders{sourceFilenameIndex} = randperm(length(openFragments{sourceFilenameIndex}.data));
            openFragmentExampleOrderIndices(sourceFilenameIndex) = 1;
            clearvars a
        end
    end

    beginMiniBatch = openFragmentExampleOrderIndices(sourceFilenameIndex);
    endMiniBatch = min(beginMiniBatch + options.miniBatchSize, ...
        length(openFragments{sourceFilenameIndex}.data));
    openFragmentExampleOrderIndices(sourceFilenameIndex) = ...
        openFragmentExampleOrderIndices(sourceFilenameIndex) + endMiniBatch;
    batchInd = openFragmentExampleOrders{sourceFilenameIndex} ...
        (beginMiniBatch:endMiniBatch);
    batch = openFragments{sourceFilenameIndex}.data(batchInd);

    % TODO: This is a hack. Remove it when the preloaded DenotationGraph data is refreshed.
    if length(batch(1).relation) < length(hyperParams.numRelations)
        if (~isempty(strfind(trainingData{sourceFilenameIndex}, 'denotation')) || ~isempty(strfind(trainingData{sourceFilenameIndex}, 'Flickr')))
            parfor i = 1:length(batch)
                if batch(i).relation == 1
                    batch(i).relation = [0 1];
                else
                    batch(i).relation = [0 2]; 
                end
            end
        else
            parfor i = 1:length(batch)
                batch(i).relation = [batch(i).relation 0];
            end
        end
    end

    [ cost, grad ] = CostGradFunc(modelState.theta, modelState.thetaDecoder, batch, modelState.constWordFeatures, hyperParams);
    modelState.sumSqGrad = modelState.sumSqGrad + grad.^2;

    % Do an AdaGrad-scaled parameter update
    adaEps = 0.001;
    modelState.theta = modelState.theta - modelState.lr * (grad ./ (sqrt(modelState.sumSqGrad) + adaEps));
    modelState.step = modelState.step + 1;
    modelState.lastHundredCosts(mod(modelState.step, 100) + 1) = cost(1);

    modelState = TestAndLog(CostGradFunc, modelState, options, trainingData, ...
        hyperParams, testDatasets);
end

end

function sourceFilenameIndex = chooseDataset(trainingData, fragmentOrderIndices, fragmentOrders, sourceSizes)
% Decide which source this minibatch will come from
% TODO: Mixing at the within-minibatch level

totalSizeOfActiveDatasets = 0;
for sourceFilenameIndex = 1:length(trainingData)
    if fragmentOrderIndices(sourceFilenameIndex) <= length(fragmentOrders{sourceFilenameIndex})
        totalSizeOfActiveDatasets = totalSizeOfActiveDatasets + sourceSizes(sourceFilenameIndex);
    end
end

if totalSizeOfActiveDatasets == 0
    sourceFilenameIndex = 0;
    return
end

randomExampleIndex = randi([1, totalSizeOfActiveDatasets],1,1);

accumulator = 0;
for sourceFilenameIndex = 1:length(trainingData)
    if fragmentOrderIndices(sourceFilenameIndex) <= length(fragmentOrders{sourceFilenameIndex})
        if randomExampleIndex <= accumulator + sourceSizes(sourceFilenameIndex)
            return
        else
            accumulator  = accumulator + sourceSizes(sourceFilenameIndex);
        end         
    end
end

disp('Error')

end
