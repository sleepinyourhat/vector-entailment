% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, embGrad, acc, confusion ] = ComputeFullCostAndGrad(theta, decoder, data, separateWordFeatures, hyperParams, computeGrad)
% Compute cost, gradient, accuracy, and confusions over a set of examples for some parameters.

B = length(data);

argout = nargout;
if nargout > 4
    confusions = zeros(B, 2);
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
    % adjacent examples are not guaranteed to share structures.

    % Accumulate log messages in memory, to avoid file corruption from
    % multiple writes within the paralellized loop.
    logMessages = cell(B, 1);

    parfor b = 1:B
        % assert(~isempty(data(b).relation), 'Null relation.')

        [localCost, localGrad, localEmbGrad, localPred] = ...
            ComputeCostAndGrad(theta, decoder, data(b), separateWordFeatures, hyperParams, computeGrad);
        accumulatedCost = accumulatedCost + localCost;
        accumulatedGrad = accumulatedGrad + localGrad;
        if hyperParams.fastEmbed
            accumulatedSeparateWordFeatureGradients = accumulatedSeparateWordFeatureGradients + localEmbGrad;
        end

        localCorrect = localPred == data(b).relation(find(data(b).relation > 0));

        if (~localCorrect) && (argout > 2) && hyperParams.showExamples
            logMessages{b} = ['for: ', data(b).left.getText(), ' ', data(b).right.getText(), ...
                  ' hypothesis:  ', num2str(localPred), ' cost: ', num2str(localCost)];
        end

        % Record statistics
        if argout > 4
            confusions(b,:) = [localPred, data(b).relation(find(data(b).relation > 0))];
        end
        accumulatedSuccess = accumulatedSuccess + localCorrect;
    end

    % Flush the accumulated log messages from inside the loop
    for b = 1:B
        if ~isempty(logMessages{b})
            Log(hyperParams.examplelog, logMessages{b});
        end
    end
    
    % Create the confusion matrix
    if nargout > 4
        confusion = zeros(hyperParams.numRelations(find(data(1).relation)));
        for b = 1:B
            confusion(confusions(b,1), confusions(b,2)) = ...
                confusion(confusions(b,1), confusions(b,2)) + 1;
        end
    end
else
    % Just compute the cost, parallelizing as above
    parfor b = 1:B
        localCost = ...
            ComputeCostAndGrad(theta, decoder, data(b), separateWordFeatures, hyperParams, 0);
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
    acc = (accumulatedSuccess / B);
end

end
