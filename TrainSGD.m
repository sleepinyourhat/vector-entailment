% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta ] = TrainSGD(CostGradFunc, modelState, options, trainingData, ...
    hyperParams, testDatasets)
% Home-baked implementation of SGD with AdaGrad/AdaDelta. Should be launched
% using TrainModel.m.

if modelState.step == 0
    Log(hyperParams.examplelog, 'Initializing SGD.')
    modelState.bestTestAcc = [0 0];
    modelState.lr = options.lr;
    modelState.pass = 0;
    modelState.lastHundredCosts = zeros(100, 1);
end 

while modelState.pass < options.numPasses
    modelState = TestAndLog(CostGradFunc, modelState, options, trainingData, ...
        hyperParams, testDatasets);

    if hyperParams.fragmentData
        modelState = TrainOnFragmentedData(CostGradFunc, trainingData, testDatasets, modelState, hyperParams, options);
    else
        modelState = TrainOnDataset(CostGradFunc, trainingData, testDatasets, modelState, hyperParams, options);
    end
        
    % Reset the AdaGrad stored weights.
    if mod(modelState.step + 1, options.resetSumSqFreq) == 0
        modelState.sumSqGrad = zeros(size(modelState.theta));
        modelState.embSumSqEmbGrad = zeros(size(modelState.separateWordFeatures));
    end

    modelState.pass = modelState.pass + 1;
end

theta = modelState.theta;

end
