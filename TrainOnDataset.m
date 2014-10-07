function [ modelState ] = TrainOnDataset(CostGradFunc, trainingData, testDatasets, modelState, hyperParams, options, filename)

N = length(trainingData);
numBatches = ceil(N/options.miniBatchSize);
randomOrder = randperm(N);

for batchNo = 0:(numBatches-1)
    beginMiniBatch = (batchNo * options.miniBatchSize+1);
    endMiniBatch = min((batchNo+1) * options.miniBatchSize,N);
    batchInd = randomOrder(beginMiniBatch:endMiniBatch);
    batch = trainingData(batchInd);

    if length(batch(1).relation) < length(hyperParams.numRelations)
        disp('Temporary hack: Modifying labels')
        if (~isempty(strfind(trainingData{sourceFilenameIndex}, 'denotation')) || ~isempty(strfind(trainingData{sourceFilenameIndex}, 'flickr')))
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