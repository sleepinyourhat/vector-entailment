% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, embGrad, acc, connectionAcc, confusion ] = ComputeBatchSentimentCostAndGrad(theta, decoder, data, separateWordFeatures, hyperParams, computeGrad)
% Compute cost, gradient, accuracy, and confusions over a batch of examples for some parameters.
% This is a well-behaved costGradFn, and can be @-passed to optimizers, including minFunc and TrainSGD.

% NOTE: This is reasonably well optimized. The time complexity here lies almost entirely within the batch objects in normal cases.

B = length(data);  % Batch size.

% Unpack theta
[ ~, ~, ...
    softmaxMatrix, trainedWordFeatures, connectionMatrix, ...
    compositionMatrix, classifierExtraMatrix, embeddingTransformMatrix] ...
    = stack2param(theta, decoder);
if hyperParams.trainWords && ~hyperParams.fastEmbed
    wordFeatures = trainedWordFeatures;
else
    wordFeatures = separateWordFeatures;
end

% Create and batch objects and run them forward.
if hyperParams.useLattices
    batch = LatticeBatch.makeLatticeBatch([data(:).sentence], wordFeatures, hyperParams);
else
    batch = SequenceBatch.makeSequenceBatch([data(:).sentence], wordFeatures, hyperParams);
end

[ features, connectionCosts, connectionAcc ] = ...
    batch.runForward(embeddingTransformMatrix, connectionMatrix, compositionMatrix, hyperParams, computeGrad);

% Set up and run top dropout.
[ features, mask ] = Dropout(features, hyperParams.topDropout, computeGrad);

extraClassifierLayerInputs = zeros(hyperParams.penultDim, B, hyperParams.topDepth);
extraClassifierLayerInnerOutputs = zeros(hyperParams.penultDim, B, hyperParams.topDepth - 1);
extraClassifierLayerInputs(:, :, 1) = features;
for layer = 1:(hyperParams.topDepth - 1) 
    extraClassifierLayerInnerOutputs(:, :, layer) = classifierExtraMatrix(:, :, layer) * [ones(1, B); extraClassifierLayerInputs(:, :, layer)];
    extraClassifierLayerInputs(:, :, layer + 1) = hyperParams.classNL(extraClassifierLayerInnerOutputs(:, :, layer));
end

[ topLabelProbs, topCosts ] = ComputeSoftmaxLayer(extraClassifierLayerInputs(:, :, hyperParams.topDepth), ...
              softmaxMatrix, hyperParams, [data(:).topLabel]');

% Sum the log losses from the three sources over all of the batch elements and normalize.
% TODO: Is it worth scaling the two different types of cost?
normalizedCost = sum([topCosts; connectionCosts]) / length(data);

% Apply regularization to the cost (does not include fastEmbed embeddings).
if hyperParams.norm == 2
    % Apply L2 regularization
    regCost = hyperParams.lambda/2 * sum(theta.^2);
else
    % Apply L1 regularization
    regCost = hyperParams.lambda * sum(abs(theta)); 
end
combinedCost = normalizedCost + regCost;

% minFunc needs a single scalar cost, not the triple that is computed here.
if ~hyperParams.minFunc
    cost = [combinedCost normalizedCost regCost];
else
    cost = combinedCost;
end

% Compute and report statistics.
accumulatedSuccess = 0;
[ ~, preds ] = max(topLabelProbs);

confusion = zeros(hyperParams.numRelations);
for b = 1:B
    localCorrect = preds(b) == data(b).topLabel(1);
    accumulatedSuccess = accumulatedSuccess + localCorrect;

    if (~localCorrect) && (nargout > 2) && hyperParams.showExamples
        Log(hyperParams.examplelog, ['for: ', data(b).sentence.getText(), ...
              ' hyp:  ', num2str(preds(b))]);
    end

    if nargout > 5
        confusion(preds(b), data(b).topLabel(1)) = ...
          confusion(preds(b), data(b).topLabel(1)) + 1;
    end
end
acc = (accumulatedSuccess / B);

if (nargin < 6 || computeGrad) && nargout > 1
    computeGrad = 1;
else
    computeGrad = 0;
    grad = [];
    embGrad = [];
end

% Compute the gradients.
if computeGrad
    [ localSoftmaxGradient, softmaxDelta ] = ...
        ComputeSoftmaxClassificationGradients(...
          softmaxMatrix, topLabelProbs, [data(:).topLabel]', extraClassifierLayerInputs(:, :, hyperParams.topDepth), hyperParams);
    localSoftmaxGradient = sum(localSoftmaxGradient, 3);

    [ localExtraMatrixGradients, extraDelta ] = ...
      ComputeExtraClassifierGradients(classifierExtraMatrix,...
          softmaxDelta, extraClassifierLayerInputs, hyperParams.classNLDeriv);

    deltaDown = extraDelta .* mask;

    [ localWordFeatureGradients, ...
      localConnectionMatrixGradients, ...
      localCompositionMatrixGradients, ...
      localEmbeddingTransformMatrixGradients ] = ...
       batch.getGradient(deltaDown, wordFeatures, embeddingTransformMatrix, ...
                            connectionMatrix, compositionMatrix, hyperParams);

    % Pack up gradients
    if hyperParams.fastEmbed
        grad = param2stack([], [], ...
            localSoftmaxGradient, ...
            [], localConnectionMatrixGradients, ...
            localCompositionMatrixGradients, ...
            localExtraMatrixGradients, ...
            localEmbeddingTransformMatrixGradients);
    else
        grad = param2stack([], [], localSoftmaxGradient, ...
            localWordFeatureGradients, localConnectionMatrixGradients, ...
            localCompositionMatrixGradients, ...
            localExtraMatrixGradients, ...
            localEmbeddingTransformMatrixGradients); 
    end

    % Normalize the gradient
    grad = grad / length(data);

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
        embGrad = localWordFeatureGradients * 1/length(data);

        for wordInd = find(embGrad(1,:))'   % TODO: Parallelize
            % Apply regularization to the gradient
            if hyperParams.norm == 2
                % Apply L2 regularization to the gradient
                embGrad(:, wordInd) = embGrad(:, wordInd) + ...
                    hyperParams.lambda * separateWordFeatures(:, wordInd);
            else
                % Apply L1 regularization to the gradient
                embGrad(:, wordInd) = embGrad(:, wordInd) + ...
                    hyperParams.lambda * sign(separateWordFeatures(:, wordInd));
            end
        end
    else
        embGrad = [];
    end

    if sum(isnan(grad)) > 0
        [ ~, ~, ...
            softmaxMatrix, trainedWordFeatures, compositionMatrices,...
            compositionMatrix, classifierExtraMatrix, ...
            embeddingTransformMatrix ] ...
            = stack2param(grad, decoder);
        
        softmaxMatrix, trainedWordFeatures, compositionMatrices, ...
        compositionMatrix, classifierExtraMatrix, ...
        embeddingTransformMatrix
        assert(false, 'NANs in computed gradient.');
    end

    assert(sum(isinf(grad)) == 0, 'Infs in computed gradient.'); 
end


end
