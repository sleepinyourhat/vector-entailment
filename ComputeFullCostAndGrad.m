% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, acc, confusion ] = ComputeFullCostAndGrad( theta, decoder, data, hyperParams, ~ )
% Compute gradient and cost with regularization over a set of examples
% for some parameters.

N = length(data);

argout = nargout;
if nargout > 3
    confusions = zeros(N, 2);
end

accumulatedCost = 0;
accumulatedSuccess = 0;
if nargout > 1
    accumulatedGrad = zeros(length(theta), 1);
end

% Check that we are set up for parallelization
if matlabpool('size') == 0 % checking to see if my pool is already open
    matlabpool;
end

if nargout > 1
    % Iterate over individual examples, letting MATLAB distribute different
    % examples to different threads.
    % Note: A single thread can only work on one example at once, since 
    % adjacent examples are not guaranteed to share trees structures.
    parfor i = 1:N
        if ~isempty(data(i).relation)
            [localCost, localGrad, localPred] = ...
                ComputeCostAndGrad(theta, decoder, data(i), hyperParams);
            accumulatedCost = accumulatedCost + localCost;
                accumulatedGrad = accumulatedGrad + localGrad;
            
            localCorrect = localPred == data(i).relation;
            if (~localCorrect) && (argout > 2) && hyperParams.showExamples
                Log(hyperParams.examplelog, ['for: ', data(i).leftTree.getText, ' ', ...
                      hyperParams.relations{data(i).relation}, ' ', ... 
                	  data(i).rightTree.getText, ...
                      ' hypothesis:  ', hyperParams.relations{localPred}]);
            end

            % Record statistics
            if argout > 3
                confusions(i,:) = [localPred, data(i).relation];
            end
            accumulatedSuccess = accumulatedSuccess + localCorrect;
        else
            Log(hyperParams.statlog, 'Bad example.');
            if argout > 3
                confusions(i,:) = [1, 1];
            end
        end

    end
    
    % Create the confusion matrix
    if nargout > 3
        confusion = zeros(hyperParams.numDataRelations);
        for i = 1:N
           confusion(confusions(i,1), confusions(i,2)) = ...
               confusion(confusions(i,1), confusions(i,2)) + 1;
        end
    end
else
    % Just compute the cost, parallelizing as above
    parfor i = 1:N
        localCost = ...
            ComputeCostAndGrad(theta, decoder, data(i), hyperParams);
        accumulatedCost = accumulatedCost + localCost;
    end
end

% Compute mean cost
normalizedCost = (1/length(data) * accumulatedCost);

if hyperParams.norm == 2
    % Apply L2 regularization
    regCost = hyperParams.lambda/2 * sum(theta.^2);
else
    % Apply L1 regularization
    regCost = hyperParams.lambda * sum(abs(theta)); 
end
combinedCost = normalizedCost + regCost;

cost = [combinedCost normalizedCost regCost]; 
% Note: Uncomment this line to use minFunc gradient checking:
% cost = combinedCost;

if nargout > 1
    % Compile the gradient
    grad = (1/length(data) * accumulatedGrad);
    if hyperParams.norm == 2
        % Apply L2 regularization to the gradient
        grad = grad + hyperParams.lambda * theta;
    else
        % Apply L1 regularization to the gradient
        grad = grad + hyperParams.lambda * sign(theta);
    end
    acc = (accumulatedSuccess / N);
end

end