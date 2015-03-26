% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ features, rnnInnerActivations ] = RunPyramidBatchForward(theta, decoder, data, separateWordFeatures, hyperParams, computeGrad)
% Compute cost, gradient, accuracy, and confusions over a set of examples for some parameters.

NUMACTIONS = 3;

B = length(data);
D = hyperParams.dim;

%%% DO EVERYTHING FOR LEFT %%%
% Factor out shortly. %

% Find the length of the longest sequence. We use this to set the size of the main feature matrix,
% to this value has a large impact on the run time of the batch.
N = 0;
for b = 1:B
    N = max([N, data(b).left.wordCount])
end

features = zeros(N, N * D, B);
rnnInnerActivations = zeros(N - 1, (N - 1) * D, B);
connectionInnerActivations = zeros(N - 1, N - 1, NUMACTIONS, B);
connections = zeros(N - 1, N - 1, NUMACTIONS, B);  % The above after softmax
connectionLabels = zeros(N - 1, N - 1, B);

if nargout > 4
    confusions = zeros(N, 2);
end

% Load the examples into the batch, including setting the bottom layer features
for b = 1:B
    for w = 1:data(b).left.wordCount
        if length(embeddingTransformMatrix) == 0
            % We have no transform layer, so just use the word features.
            obj.features(N, (w - 1) * D + 1:w * D, b) = wordFeatures(data(b).left.wordIndices(w)), :)'; 
        else
            % TODO
            assert(false, 'Not implemented.')
        end
    end
    connectionLabels(N - data(b).left.wordCount + 1:N, 1:data(b).left.wordCount, b) = data(b).left.connectionLabels;
end 

% Load word vectors into features
% Run main loop:
%% Make decision
%% Compose
%% Combine

% Gradient!


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
