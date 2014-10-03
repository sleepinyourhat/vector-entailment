function modelState = TestAndLog(CostGradFunc, modelState, options, trainingData, ...
    hyperParams, testDatasets)

% TODO: Change freqs

% Do mid-run testing
if mod(modelState.step, options.testFreq) == 0

    % Test on training data
    cost = mean(modelState.lastHundredCosts(1:min(modelState.step, 100)));
    acc = -1;
    if ~hyperParams.fragmentData
        if mod(modelState.step, options.examplesFreq) == 0 && modelState.step > 0
            hyperParams.showExamples = true;
            Log(hyperParams.examplelog, 'Training data:')
        else
            hyperParams.showExamples = false;
        end
        [cost, ~, acc] = CostGradFunc(modelState.theta, modelState.thetaDecoder, trainingData, modelState.constWordFeatures, hyperParams);
    end

    % Test on test data
    if nargin > 5
        if mod(modelState.step, options.confusionFreq) == 0 && modelState.step > 0
            hyperParams.showConfusions = true;
        else
            hyperParams.showConfusions = false;
        end

        if (mod(modelState.step, options.examplesFreq) == 0 || mod(modelState.step, options.confusionFreq) == 0) && modelState.step > 0
            Log(hyperParams.statlog, 'Test data:');
        end
        testAcc = TestModel(CostGradFunc, modelState.theta, modelState.thetaDecoder, testDatasets, modelState.constWordFeatures, hyperParams);
        modelState.bestTestAcc = max(testAcc, modelState.bestTestAcc);
    else
        testAcc = -1;
    end

    % Log statistics
    if testAcc ~= -1
        Log(hyperParams.statlog, ['pass ', num2str(modelState.pass), ' step ', num2str(modelState.step), ...
            ' train acc: ', num2str(acc), ' test acc: ', num2str(testAcc), ' (best: ', ...
            num2str(modelState.bestTestAcc), ')']);
    else
        Log(hyperParams.statlog, ['pass ', num2str(modelState.pass), ' step ', num2str(modelState.step), ...
            ' acc: ', num2str(acc)]);
    end
elseif mod(modelState.step, options.costFreq) == 0
    % Just compute the cost.
    cost = mean(modelState.lastHundredCosts(1:min(modelState.step, 100)));
end

if mod(modelState.step, options.checkpointFreq) == 0
    % Write a checkpoint to disk.
    save([options.name, '/', 'ckpt-', options.runName, datestr(now, 'yymmddHHMMSS'),...
       '@', num2str(modelState.step)] , 'modelState');
end

% Log the cost.
if mod(modelState.step, options.costFreq) == 0
    Log(hyperParams.statlog, ['pass ', num2str(modelState.pass), ' step ', num2str(modelState.step), ...
        ' cost: ', num2str(cost)]);
end

end
