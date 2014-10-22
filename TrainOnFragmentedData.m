function [ modelState ] = TrainOnFragmentedData(CostGradFunc, trainingData, testDatasets, modelState, hyperParams, options)

% Preloaded MAT files for each training dataset, 
% outer index = sourceFilenameIndex, inner index reflect
% an arbitrary order within the framents of the source file.
openFragments = {}; % Loaded data by filenameIndex.
fragmentFiles = {}; % 
fragmentOrders = {}; % Permutations of the fragment indices for each 
fragmentOrderIndices = []; % How far along training is in the list of fragments for each file.
sourceSizes = []; % The length of each complete dataset.
openFragmentExampleOrders = {}; % Permutations of the item indices in each open fragment.
openFragmentExampleOrderIndices = []; % How far along training is in each fragment.
totalNumTrainingExamples = 0;

% Initialize all of the source files.
for sourceFilenameIndex = 1:length(trainingData)

    % Find the files to use.
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

% Iterate through all of the data.
while true
    sourceFilenameIndex = chooseDataset(trainingData, fragmentOrderIndices, fragmentOrders, sourceSizes);
    if sourceFilenameIndex == 0
        disp(['Finished pass at step ' num2str(modelState.step)]);
        return
    end

    % Load a new file if needed
    if openFragmentExampleOrderIndices(sourceFilenameIndex) > length(openFragments{sourceFilenameIndex}.data)
        fragmentOrderIndices(sourceFilenameIndex) = fragmentOrderIndices(sourceFilenameIndex) + 1;
        
        % Check that we haven't just exhausted this source
        if fragmentOrderIndices(sourceFilenameIndex) > length(fragmentOrders{sourceFilenameIndex})
            continue
        else
            % Open a file and permute its examples
            nextFragmentIndex = fragmentOrders{sourceFilenameIndex}(fragmentOrderIndices(sourceFilenameIndex));
            filename = fragmentFiles{sourceFilenameIndex}{nextFragmentIndex};
            filepath = fileparts(trainingData{sourceFilenameIndex});
            disp(['Loading ' filename]);
            openFragments{sourceFilenameIndex} = load([filepath, '/', filename],'-mat');
            openFragmentExampleOrders{sourceFilenameIndex} = randperm(length(openFragments{sourceFilenameIndex}.data));
            openFragmentExampleOrderIndices(sourceFilenameIndex) = 1;
            clearvars a
        end
    end

    % Work out which span of the example order index to look in.
    beginMiniBatch = openFragmentExampleOrderIndices(sourceFilenameIndex);
    endMiniBatch = min(beginMiniBatch + options.miniBatchSize - 1, ...
        length(openFragments{sourceFilenameIndex}.data));

    % Skip updates that are too small.
    if (endMiniBatch - beginMiniBatch + 1) < options.miniBatchSize
        openFragmentExampleOrderIndices(sourceFilenameIndex) = ...
            openFragmentExampleOrderIndices(sourceFilenameIndex) + (endMiniBatch - beginMiniBatch + 1); 
        continue;
    end       

    openFragmentExampleOrderIndices(sourceFilenameIndex) = ...
        openFragmentExampleOrderIndices(sourceFilenameIndex) + (endMiniBatch - beginMiniBatch + 1);

    % Work out the actual example numbers to train on.
    batchInd = openFragmentExampleOrders{sourceFilenameIndex} ...
        (beginMiniBatch:endMiniBatch);
    batch = openFragments{sourceFilenameIndex}.data(batchInd);
    assert(length(batch) == options.miniBatchSize, 'Batch size wrong!');

    % Run the minibatch.
    [ cost, grad, embGrad ] = CostGradFunc(modelState.theta, modelState.thetaDecoder, batch, modelState.separateWordFeatures, hyperParams);

    modelState.sumSqGrad = modelState.sumSqGrad + grad.^2;

    % Do an AdaGrad-scaled parameter update
    adaEps = 0.001;
    modelState.theta = modelState.theta - modelState.lr * (grad ./ (sqrt(modelState.sumSqGrad) + adaEps));

    assert(sum(isnan(modelState.theta)) == 0, 'NaNs in theta.')
    assert(sum(isinf(modelState.theta)) == 0, 'Infs in theta.')

    % Do an AdaGrad-scaled parameter update to the separate word features
    if hyperParams.fastEmbed
        modelState.sumSqEmbGrad = modelState.sumSqEmbGrad + embGrad.^2;
        modelState.separateWordFeatures = modelState.separateWordFeatures - modelState.lr * (embGrad ./ (sqrt(modelState.sumSqEmbGrad) + adaEps));
    end

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

assert(false, 'chooseDataset failed.')

end
