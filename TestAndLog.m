function modelState = TestAndLog(CostGradFunc, modelState, options, trainingData, hyperParams, testDatasets)
% Do a full set of test runs.

if mod(modelState.step, options.testFreq) == 0
    Log(hyperParams.statlog, ['Theta min/mean/meanabs/max: ',  num2str(min(modelState.theta)), ' ', ...
                                          num2str(mean(modelState.theta)), ' ', ...
                                          num2str(mean(abs(modelState.theta))), ' ', ...
                                          num2str(max(modelState.theta))]);

    Log(hyperParams.statlog, ['Word min/mean/meanabs/max: ',  num2str(min(modelState.separateWordFeatures(:))), ' ', ...
                                      num2str(mean(modelState.separateWordFeatures(:))), ' ', ...
                                      num2str(mean(abs(modelState.separateWordFeatures(:)))), ' ', ...
                                      num2str(max(modelState.separateWordFeatures(:)))]);

    % Test on training data.
    cost = mean(modelState.lastHundredCosts(1:min(modelState.step, 100)));
    acc = -1;
    conAcc = [ -1; -1 ];
    macro = -1;
    if ~hyperParams.fragmentData
        if length(trainingData) > hyperParams.maxTrainingEvalSampleSize
            randomOrder = randperm(length(trainingData));
            trainingSample = trainingData(randomOrder(1:hyperParams.maxTrainingEvalSampleSize));
        else
            trainingSample = trainingData;
        end

        if mod(modelState.step, options.detailedStatFreq) == 0
            hyperParams.showDetailedStats = true;
        else
            hyperParams.showDetailedStats = false;
        end

        if mod(modelState.step, options.examplesFreq) == 0 && modelState.step > 0
            hyperParams.showExamples = true;
            Log(hyperParams.examplelog, 'Training data:')
        else
            hyperParams.showExamples = false;
        end

        if length(hyperParams.numLabels) == 1
            [cost, ~, ~, acc, conAcc, conf] = CostGradFunc(modelState.theta, modelState.thetaDecoder, trainingSample, modelState.separateWordFeatures, hyperParams, 0);
            macro = GetMacroF1(conf);
        else
            [cost, ~, ~, acc, conAcc] = CostGradFunc(modelState.theta, modelState.thetaDecoder, trainingSample, modelState.separateWordFeatures, hyperParams, 0);
        end
    end

    % Test on test data.
    if nargin > 5
        if mod(modelState.step, options.detailedStatFreq) == 0
            hyperParams.showDetailedStats = true;
        else
            hyperParams.showDetailedStats = false;
        end

        if mod(modelState.step, options.examplesFreq) == 0 && modelState.step > 0
            hyperParams.showExamples = true;
        else
            hyperParams.showExamples = false;
        end

        if (mod(modelState.step, options.examplesFreq) == 0 || mod(modelState.step, options.detailedStatFreq) == 0) && modelState.step > 0
            Log(hyperParams.examplelog, 'Test data:');
        end
        [testAcc, testMf1, ~, testConAcc] = TestModel(CostGradFunc, modelState.theta, modelState.thetaDecoder, testDatasets, modelState.separateWordFeatures, hyperParams);
        modelState.bestTestAcc = max(testAcc, modelState.bestTestAcc);
        hyperParams.showExamples = false;
        if (testAcc(1) == modelState.bestTestAcc(1)) && (modelState.step > 0)
            % Write a checkpoint to disk.
            delete([options.name, '/ckpt-best-', options.runName, '*']);
            save([options.name, '/ckpt-best-', options.runName, datestr(now, 'yymmddHHMMSS'), ...
                    '@', num2str(modelState.step)] , 'modelState');
        end
    else
        testAcc = -1;
    end

    % Log statistics.
    if testAcc ~= -1
        Log(hyperParams.statlog, ['pass ', num2str(modelState.pass), ' step ', num2str(modelState.step), ...
            ' train acc: ', num2str(acc), ' (mf1 ', num2str(macro), ') test acc: ', num2str(testAcc), ...
            ' mf1: ', num2str(testMf1), ' (best: ', num2str(modelState.bestTestAcc), ')']);
    else
        Log(hyperParams.statlog, ['pass ', num2str(modelState.pass), ' step ', num2str(modelState.step), ...
            ' acc: ', num2str(acc)]);
    end

    if conAcc(1) ~= -1
        Log(hyperParams.statlog, ['pass ', num2str(modelState.pass), ' step ', num2str(modelState.step), ...
            ' train connection acc: ', num2str(conAcc(1, :)), ' std ', num2str(conAcc(2, :)), ...
            ' test connection acc: ',  num2str(testConAcc(1, :)), ' std ', num2str(testConAcc(2, :))]);
    end

    FlushLogs(hyperParams);
elseif mod(modelState.step, options.costFreq) == 0
    % Just compute the cost.
    cost = mean(modelState.lastHundredCosts(1:min(modelState.step, 100)));
end

if mod(modelState.step, options.checkpointFreq) == 0 && modelState.step > 0
    % Write a checkpoint to disk.
    cModelState = modelState
    cModelState
    save([options.name, '/', 'ckpt-', options.runName, datestr(now, 'yymmddHHMMSS'),...
       '@', num2str(modelState.step)], 'cModelState');
end

% Log the cost.
if mod(modelState.step, options.costFreq) == 0
    Log(hyperParams.statlog, ['pass ', num2str(modelState.pass), ' step ', num2str(modelState.step), ...
        ' cost: ', num2str(cost)]);
    FlushLogs(hyperParams);
end

end
