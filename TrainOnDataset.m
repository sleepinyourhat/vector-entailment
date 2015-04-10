% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ modelState ] = TrainOnDataset(CostGradFunc, trainingData, testDatasets, modelState, hyperParams, options, filename)
% Run one SGD pass through an in-memory dataset.

% Shuffle the data, using multiple copies of some examples if requested.
if isfield(hyperParams, 'firstMultiplier')
    modifiedOrder = hyperParams.firstCutoff + 1:length(trainingData);
    for i = 1:hyperParams.firstMultiplier
        modifiedOrder = [modifiedOrder 1:hyperParams.firstCutoff];
    end
    tempOrder = randperm(length(modifiedOrder));
    randomOrder = modifiedOrder(tempOrder);
elseif isfield(hyperParams, 'trainingMultipliers')
    modifiedOrder = [];
    startIndex = 1;
    for datasetIndex = 1:length(hyperParams.trainingMultipliers)
        for count = 1:hyperParams.trainingMultipliers(datasetIndex)
            modifiedOrder = [modifiedOrder, startIndex:startIndex + hyperParams.trainingLengths(datasetIndex) - 1];
        end
        startIndex = startIndex + hyperParams.trainingLengths(datasetIndex);
    end
    tempOrder = randperm(length(modifiedOrder));
    randomOrder = modifiedOrder(tempOrder);
    assert(length(randomOrder) == sum(hyperParams.trainingLengths .* hyperParams.trainingMultipliers));
else
    randomOrder = randperm(length(trainingData));
end

N = length(randomOrder); % Number of examples in full dataset.
numBatches = ceil(N/options.miniBatchSize);

for batchNo = 0:(numBatches - 1)
    beginMiniBatch = (batchNo * options.miniBatchSize + 1);
    endMiniBatch = (batchNo + 1) * options.miniBatchSize;

    % Don't bother with the last few examples if they don't make up a full minibatch. 
    % They'll be reshuffled in the next epoch.
    if endMiniBatch > N
        return
    end

    batchInd = randomOrder(beginMiniBatch:endMiniBatch);
    batch = trainingData(batchInd);

    [ cost, grad, embGrad ] = CostGradFunc(modelState.theta, modelState.thetaDecoder, batch, modelState.separateWordFeatures, hyperParams, true);

    modelState = options.updateFn(modelState, options, grad, embGrad);

    modelState.step = modelState.step + 1;
    modelState.lastHundredCosts(mod(modelState.step, 100) + 1) = cost(1);

    modelState = TestAndLog(CostGradFunc, modelState, options, trainingData, ...
        hyperParams, testDatasets);
end

end
