% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta ] = AdaGradSGD(CostGradFunc, modelState, options, trainingData, ...
    hyperParams, testDatasets)
% Home-baked implementation of SGD with AdaGrad.

N = length(trainingData);

if modelState.pass == 0
    modelState.prevCost = intmax;
    modelState.bestTestErr = 1;
end 

lr = options.lr;

modelState.sumSqGrad = zeros(size(modelState.theta));

while modelState.pass < options.numPasses

    % Do mid-run testing
    if mod(modelState.pass, options.testFreq) == 0
        
        % Test on training data
        if mod(modelState.pass, options.examplesFreq) == 0 && modelState.pass > 0
            hyperParams.showExamples = true;
            Log(hyperParams.examplelog, 'Training data:')
        else
            hyperParams.showExamples = false;
        end
        [cost, ~, acc] = CostGradFunc(modelState.theta, modelState.thetaDecoder, trainingData, hyperParams);
        
        % Test on test data
        if nargin > 5
            if mod(modelState.pass, options.confusionFreq) == 0 && modelState.pass > 0
                hyperParams.showConfusions = true;
            else
                hyperParams.showConfusions = false;
            end

            if (mod(modelState.pass, options.examplesFreq) == 0 || mod(modelState.pass, options.confusionFreq) == 0) && modelState.pass > 0
                Log(hyperParams.statlog, 'Test data:');
            end
            testErr = TestModel(CostGradFunc, modelState.theta, modelState.thetaDecoder, testDatasets, hyperParams);
            modelState.bestTestErr = min(testErr, modelState.bestTestErr);
        else
            testErr = -1;
        end

        % Log statistics
        if testErr ~= -1
            Log(hyperParams.statlog, ['pass ', num2str(modelState.pass), ' train PER: ', num2str(acc), ...
                  ' test PER: ', num2str(testErr), ' (best: ', ...
                  num2str(modelState.bestTestErr), ')']);
        else
            Log(hyperParams.statlog, ['pass ', num2str(modelState.pass), ' PER: ', num2str(acc)]);
        end
    else
        % Just compute the cost.
       cost = CostGradFunc(modelState.theta, modelState.thetaDecoder, trainingData, hyperParams);
    end
    if mod(modelState.pass, options.checkpointFreq) == 0
        % Write a checkpoint to disk.
        % TODO: Use integer timestamp for sorting.
        save([options.name, '/', 'ckpt-', options.runName, datestr(now, 'yymmddHHMMSS'),...
           '@', num2str(modelState.pass)] , 'modelState');
    end

    % Log the cost.
    Log(hyperParams.statlog, ['pass ', num2str(modelState.pass), ' cost: ', num2str(cost)]);

    % Check the stopping criterion.
    if abs(modelState.prevCost - cost(1)) < 10e-7
        Log(hyperParams.statlog, 'Stopped improving.');
        break;
    end

    modelState.prevCost = cost(1);
    numBatches = ceil(N/options.miniBatchSize);
    randomOrder = randperm(N);

    % Train.
    for batchNo = 0:(numBatches-1)
        beginMiniBatch = (batchNo * options.miniBatchSize+1);
        endMiniBatch = min((batchNo+1) * options.miniBatchSize,N);
        batchInd = randomOrder(beginMiniBatch:endMiniBatch);
        batch = trainingData(batchInd);
        [ ~, grad ] = CostGradFunc(modelState.theta, modelState.thetaDecoder, batch, hyperParams);
        modelState.sumSqGrad = modelState.sumSqGrad + grad.^2;

        % Do an AdaGrad-scaled parameter update
        adaEps = 0.001;
        modelState.theta = modelState.theta - lr * (grad ./ (sqrt(modelState.sumSqGrad) + adaEps));
    end

    % Reset the AdaGrad stored weights.
    if mod(modelState.pass + 1, options.resetSumSqFreq) == 0
        modelState.sumSqGrad = zeros(size(modelState.theta));
    end

    modelState.pass = modelState.pass + 1;
end

end
