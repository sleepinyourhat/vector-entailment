% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, embGrad, acc, confusion ] = ComputeFullCostAndGrad( theta, decoder, data, separateWordFeatures, hyperParams, computeGrad)
% Compute cost, gradient, accuracy, and confusions over a set of examples for some parameters.

N = length(data);

argout = nargout;
if nargout > 4
    confusions = zeros(N, 2);
end

accumulatedCost = 0;
accumulatedSuccess = 0;
if (nargin < 6 || computeGrad) && nargout > 1
    computeGrad = 1;
    accumulatedGrad = zeros(length(theta), 1);

    % If fastEmbed is on, set up a separate sparse accumulator for the embeddings.
    if hyperParams.fastEmbed
        accumulatedSeparateWordFeatureGradients = sparse([], [], [], ...
          size(separateWordFeatures, 1), size(separateWordFeatures, 2), hyperParams.embeddingDim * 5 * length(data));
        % TODO: This unsparsifies below. Investigate.
    else
        accumulatedSeparateWordFeatureGradients = [];
    end
else
    computeGrad = 0;
    accumulatedSeparateWordFeatureGradients = [];
    accumulatedGrad = [];
    grad = [];
    embGrad = [];
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
        % assert(~isempty(data(i).relation), 'Null relation.')

        [localCost, localGrad, localEmbGrad, localPred] = ...
            ComputeCostAndGrad(theta, decoder, data(i), separateWordFeatures, hyperParams, computeGrad);
        accumulatedCost = accumulatedCost + localCost;
        accumulatedGrad = accumulatedGrad + localGrad;
        if hyperParams.fastEmbed
            accumulatedSeparateWordFeatureGradients = accumulatedSeparateWordFeatureGradients + localEmbGrad;
        end
        
        localCorrect = localPred == data(i).relation(find(data(i).relation > 0));

        if (~localCorrect) && (argout > 2) && hyperParams.showExamples
            logMessages{i} = ['for: ', data(i).leftTree.getText, ' ', data(i).rightTree.getText, ...
                  ' hypothesis:  ', num2str(localPred), ' cost: ', num2str(localCost)];
        end

        % Record statistics
        if argout > 4
            confusions(i,:) = [localPred, data(i).relation(find(data(i).relation > 0))];
        end
        accumulatedSuccess = accumulatedSuccess + localCorrect;
    end

    % Flush the accumulated log messages from inside the loop
    for i = 1:N
        if ~isempty(logMessages{i})
            Log(hyperParams.examplelog, logMessages{i});
        end
    end
    
    % Create the confusion matrix
    if nargout > 4
        confusion = zeros(hyperParams.numRelations(find(data(1).relation)));
        for i = 1:N
            confusion(confusions(i,1), confusions(i,2)) = ...
                confusion(confusions(i,1), confusions(i,2)) + 1;
        end
    end
else
    % Just compute the cost, parallelizing as above
    parfor i = 1:N
        localCost = ...
            ComputeCostAndGrad(theta, decoder, data(i), separateWordFeatures, hyperParams);
        accumulatedCost = accumulatedCost + localCost;
    end
end

% Compute mean cost
normalizedCost = (1/length(data) * accumulatedCost);

% Apply regularization to the cost (does not include fastEmbed embeddings).
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

if computeGrad
    % Compile the gradient
    grad = (1/length(data) * accumulatedGrad);

    % Apply regularization to the gradient
    if hyperParams.norm == 2
        % Apply L2 regularization to the gradient
        grad = grad + hyperParams.lambda * theta;
    else
        % Apply L1 regularization to the gradient
        grad = grad + hyperParams.lambda * sign(theta);
    end

    if hyperParams.fastEmbed
        % Compile the embedding gradient
        embGrad = accumulatedSeparateWordFeatureGradients * 1/length(data);

        for wordInd = find(embGrad(:,1))'   % TODO: Parallelize
            % Apply regularization to the gradient
            if hyperParams.norm == 2
                % Apply L2 regularization to the gradient
                embGrad(wordInd, :) = embGrad(wordInd, :) + ...
                    hyperParams.lambda * separateWordFeatures(wordInd, :);
            else
                % Apply L1 regularization to the gradient
                embGrad(wordInd, :) = embGrad(wordInd, :) + ...
                    hyperParams.lambda * sign(separateWordFeatures(wordInd, :));
            end
            assert(sum(isnan(embGrad(wordInd, :))) == 0, 'NaNs in computed embedding gradient.');
            assert(sum(isinf(embGrad(wordInd, :))) == 0, 'Infs in computed embedding gradient.');
        end
    else
        embGrad = [];
    end

    assert(sum(isnan(grad)) == 0, 'NaNs in computed gradient.');
    assert(sum(isinf(grad)) == 0, 'Infs in computed gradient.'); 
end

if nargout > 3
    acc = (accumulatedSuccess / N);
end

end