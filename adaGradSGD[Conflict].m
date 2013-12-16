function [ theta ] = adaGradSGD(theta, options, thetaDecoder, data, hyperParams, testDatasets)

N = length(data);
prevCost = intmax;
bestTestErr = 1;

for pass = 1:options.numPasses
    numBatches = ceil(N/options.miniBatchSize);
    sumSqGrad = zeros(size(theta));
    randomOrder = randperm(N);
    accumulatedCost = [0, 0, 0];
    
    for batchNo = 0:(numBatches-1)
        beginMiniBatch = (batchNo * options.miniBatchSize+1);
        endMiniBatch = min((batchNo+1) * options.miniBatchSize,N);
        batchInd = randomOrder(beginMiniBatch:endMiniBatch);
        batch = data(batchInd);
        [ cost, grad ] = ComputeFullCostAndGrad(theta, thetaDecoder, batch, hyperParams);
        accumulatedCost = accumulatedCost + cost;
        sumSqGrad = sumSqGrad + grad.^2;
        
        % Do adaGrad update
        adaEps = 0.001;
        theta = theta - options.lr * (grad ./ (sqrt(sumSqGrad) + adaEps));
    end
    accumulatedCost = accumulatedCost / numBatches;
    disp(['pass ', num2str(pass), ' costs: ', num2str(accumulatedCost)]);

    if abs(prevCost - accumulatedCost(1)) < 10e-7
        disp('Stopped improving.');
        break;
    end
    prevCost = accumulatedCost(1);
    
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
        if nargin > 5
            if mod(pass, options.confusionFreq) == 0
                hyperParams.showConfusions = true;
            else
                hyperParams.showConfusions = false;
            end

            if mod(pass, options.examplesFreq) == 0 || mod(pass, options.confusionFreq) == 0
                disp('Test data:')
            end
            testErr = TestModel(theta, thetaDecoder, testDatasets, hyperParams);
            if bestTestErr > testErr
                bestTestErr = testErr;
            end
        else
            testErr = -1;
        end
        if testErr ~= -1
            disp(['pass ', num2str(pass), ' train PER: ', num2str(acc), ...
                  ' test PER: ', num2str(testErr), ' (best: ', ...
                  num2str(bestTestErr), ')']);
        else
            disp(['pass ', num2str(pass), ' PER: ', num2str(acc)]);
        end
    end
    if mod(pass, options.checkpointFreq) == 0
        save([options.name, '/', 'checkpoint-', options.runName, '@', ...
            num2str(pass)] , 'theta', 'thetaDecoder');
    end
        
    
end

end