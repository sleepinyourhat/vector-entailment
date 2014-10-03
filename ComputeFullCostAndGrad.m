% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, acc, confusion ] = ComputeFullCostAndGrad( theta, decoder, data, constWordFeatures, hyperParams, ~)
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

    % Accumulate log messages in memory, to avoid file corruption from
    % multiple writes within the paralellized loop.
    logMessages = cell(N, 1);

    parfor i = 1:N
        if ~isempty(data(i).relation)
            [localCost, localGrad, localPred] = ...
                ComputeCostAndGrad(theta, decoder, data(i), constWordFeatures, hyperParams);
            accumulatedCost = accumulatedCost + localCost;
                accumulatedGrad = accumulatedGrad + localGrad;
            
            localCorrect = localPred == data(i).relation;
            if (~localCorrect) && (argout > 2) && hyperParams.showExamples
                logMessages{i} = ['for: ', data(i).leftTree.getText, ' ', ...
                      hyperParams.relations{data(i).relation}, ' ', ... 
                	  data(i).rightTree.getText, ...
                      ' hypothesis:  ', hyperParams.relations{localPred}];
            end

            % Record statistics
            if argout > 3
                confusions(i,:) = [localPred, data(i).relation];
            end
            accumulatedSuccess = accumulatedSuccess + localCorrect;
        else
            disp('Bad example.');
            if argout > 3
                confusions(i,:) = [1, 1];
            end
        end

    end

    % Flush the accumulated log messages from inside the loop
    for i = 1:N
        if ~isempty(logMessages{i})
            Log(hyperParams.examplelog, logMessages{i});
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
            ComputeCostAndGrad(theta, decoder, data(i), constWordFeatures, hyperParams);
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

% minFunc needs a single scalar cost, not the triple that is reported here.
if ~hyperParams.minFunc
    cost = [combinedCost normalizedCost regCost]; 
else
    cost = combinedCost;
end

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

    size(grad)
    size(theta)

    acc = (accumulatedSuccess / N);
end


end