function [ modelState ] = TrainOnDataset(CostGradFunc, trainingData, testDatasets, modelState, hyperParams, options, filename)

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
            modifiedOrder = [modifiedOrder startIndex:startIndex + hyperParams.trainingLengths(datasetIndex) - 1];
        end
        startIndex = startIndex + hyperParams.trainingLengths(datasetIndex);
    end
    tempOrder = randperm(length(modifiedOrder));
    randomOrder = modifiedOrder(tempOrder);
else
    randomOrder = randperm(N);
end

N = length(randomOrder);
numBatches = ceil(N/options.miniBatchSize);

for batchNo = 0:(numBatches - 1)
    beginMiniBatch = (batchNo * options.miniBatchSize + 1);
    endMiniBatch = (batchNo + 1) * options.miniBatchSize;

    % Don't bother with the last few examples if they don't make up a full minibatch. 
    % They'll be reshuffled in the next pass.
    if endMiniBatch > N
        return
    end

    batchInd = randomOrder(beginMiniBatch:endMiniBatch);
    batch = trainingData(batchInd);

    [ cost, grad, embGrad ] = CostGradFunc(modelState.theta, modelState.thetaDecoder, batch, modelState.separateWordFeatures, hyperParams);

    % Do an AdaGrad-scaled parameter update
    modelState.sumSqGrad = modelState.sumSqGrad + grad.^2;
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