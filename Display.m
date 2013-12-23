% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function stop = Display(theta,~,i,~,~,~,~,~,~,~,thetaDecoder, data, hyperParams, testDatasets)
% This is passed to minFunc to get informative mid-run displays of the sort
% used in AdaGradSGD.

stop = 0;
pass = i;
global options

% Do mid-run testing:
if mod(pass, options.testFreq) == 0

    % Test on training data:
    if mod(pass, options.examplesFreq) == 0
        hyperParams.showExamples = true;
        disp('Training data:')
    else
        hyperParams.showExamples = false;
    end
    [~, ~, acc] = ComputeFullCostAndGrad(theta, thetaDecoder, data, hyperParams);

    % Test on test data:
    if nargin > 13
        if mod(pass, options.confusionFreq) == 0
            hyperParams.showConfusions = true;
        else
            hyperParams.showConfusions = false;
        end

        if mod(pass, options.examplesFreq) == 0 || mod(pass, options.confusionFreq) == 0
            disp('Test data:')
        end
        testErr = TestModel(theta, thetaDecoder, testDatasets, hyperParams);
    else
        testErr = -1;
    end
    if testErr ~= -1
        disp(['pass ', num2str(pass), ' train PER: ', num2str(acc), ...
              ' test PER: ', num2str(testErr)]);
    else
        disp(['pass ', num2str(pass), ' PER: ', num2str(acc)]);
    end
end
if mod(pass, options.checkpointFreq) == 0
    save([options.name, '/', 'pretrained-theta-wordpairs-', ...
        num2str(hyperParams.dim), 'x', ...
        num2str(hyperParams.penultDim), '-', options.runName, '@', ...
        num2str(pass)] , 'theta', 'thetaDecoder');
end

end