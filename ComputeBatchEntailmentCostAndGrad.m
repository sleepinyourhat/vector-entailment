% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, embGrad, acc, connectionAcc, confusion ] = ComputeBatchEntailmentCostAndGrad(theta, decoder, data, separateWordFeatures, hyperParams, computeGrad)
% Compute cost, gradient, accuracy, and confusions over a batch of examples for some parameters.
% This is a well-behaved costGradFn, and can be @-passed to optimizers, including minFunc and TrainSGD.

% NOTE: This is reasonably well optimized. The time complexity here lies almost entirely within the batch objects in normal cases.

B = length(data);  % Batch size.


if (nargin < 6 || computeGrad) && nargout > 1
    computeGrad = 1;
else
    computeGrad = 0;
    grad = [];
    embGrad = [];
end

% Unpack theta
[ mergeMatrices, mergeMatrix, ...
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
    leftBatch = LatticeBatch.makeLatticeBatch([data(:).left], wordFeatures, hyperParams);
    rightBatch = LatticeBatch.makeLatticeBatch([data(:).right], wordFeatures, hyperParams);
else
    leftBatch = SequenceBatch.makeSequenceBatch([data(:).left], wordFeatures, hyperParams);
    rightBatch = SequenceBatch.makeSequenceBatch([data(:).right], wordFeatures, hyperParams);
end

[ leftFeatures, leftConnectionCosts, leftConnectionAcc ] = ...
    leftBatch.runForward(embeddingTransformMatrix, connectionMatrix, compositionMatrix, hyperParams, computeGrad);
[ rightFeatures, rightConnectionCosts, rightConnectionAcc ] = ...
    rightBatch.runForward(embeddingTransformMatrix, connectionMatrix, compositionMatrix, hyperParams, computeGrad);

% TODO: Weighted average.
connectionAcc = [leftConnectionAcc, rightConnectionAcc];

% Set up and run top dropout.
[ leftFeatures, leftMask ] = Dropout(leftFeatures, hyperParams.topDropout, computeGrad);
[ rightFeatures, rightMask ] = Dropout(rightFeatures, hyperParams.topDropout, computeGrad);

% Compute classification tensor layer (or plain RNN layer).
if hyperParams.useThirdOrderMerge
    mergeOutput = ComputeTensorLayer(leftFeatures, ...
        rightFeatures, mergeMatrices, mergeMatrix, hyperParams.classNL);
else
    mergeOutput = ComputeRNNLayer(leftFeatures, rightFeatures, ...
        mergeMatrix, hyperParams.classNL);
end

% Post-merge layers
extraClassifierLayerInputs = zeros(hyperParams.penultDim, B, hyperParams.topDepth);
extraClassifierLayerInnerOutputs = zeros(hyperParams.penultDim, B, hyperParams.topDepth - 1);
extraClassifierLayerInputs(:, :, 1) = permute(mergeOutput, [1, 3, 2]);
for layer = 1:(hyperParams.topDepth - 1) 
    extraClassifierLayerInnerOutputs(:, :, layer) = classifierExtraMatrix(:, :, layer) * [ones(1, B); extraClassifierLayerInputs(:, :, layer)];
    extraClassifierLayerInputs(:, :, layer + 1) = hyperParams.classNL(extraClassifierLayerInnerOutputs(:, :, layer));
end

[ relationProbs, topCosts ] = ComputeSoftmaxLayer(extraClassifierLayerInputs(:, :, hyperParams.topDepth), ...
              softmaxMatrix, hyperParams, [data(:).relation]');

% Sum the log losses from the three sources over all of the batch elements and normalize.
% TODO: Is it worth scaling the two different types of cost?
normalizedCost = sum([topCosts; leftConnectionCosts; rightConnectionCosts]) / length(data);

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
[ ~, preds ] = max(relationProbs);
confusion = zeros(hyperParams.numRelations(data(1).relation(2)));
for b = 1:B
    localCorrect = preds(b) == data(b).relation(1);
    accumulatedSuccess = accumulatedSuccess + localCorrect;

    if (~localCorrect) && (nargout > 2) && hyperParams.showExamples
        Log(hyperParams.examplelog, ['for: ', data(b).left.getText(), ' ', data(b).right.getText(), ...
              ' hyp:  ', num2str(preds(b))]);
    end

    if nargout > 5
        confusion(preds(b), data(b).relation(1)) = ...
          confusion(preds(b), data(b).relation(1)) + 1;
    end
end

acc = (accumulatedSuccess / B);

% Compute the gradients.
if computeGrad
    [ localSoftmaxGradient, softmaxDelta ] = ...
        ComputeSoftmaxClassificationGradients(...
          softmaxMatrix, relationProbs, [data(:).relation]', extraClassifierLayerInputs(:, :, hyperParams.topDepth), hyperParams);
    localSoftmaxGradient = sum(localSoftmaxGradient, 3);

    [ localExtraMatrixGradients, extraDelta ] = ...
      ComputeExtraClassifierGradients(classifierExtraMatrix,...
          softmaxDelta, extraClassifierLayerInputs, hyperParams.classNLDeriv);

    if hyperParams.useThirdOrderMerge
        % Compute gradients for the merge tensor layer
        [ localMergeMatricesGradients, ...
            localMergeMatrixGradients, ...
            MergeDeltaLeft, MergeDeltaRight ] = ...
          ComputeTensorLayerGradients(leftFeatures, rightFeatures, ...
              mergeMatrices, mergeMatrix, ...
              extraDelta, hyperParams.classNLDeriv, mergeOutput);
          localMergeMatricesGradients = sum(localMergeMatricesGradients, 4);
    else
         % Compute gradients for the merge NN layer
         localMergeMatricesGradients = [];  
         [localMergeMatrixGradients, ...
            MergeDeltaLeft, MergeDeltaRight] = ...
          ComputeRNNLayerGradients(leftFeatures, rightFeatures, ...
              mergeMatrix, extraDelta, hyperParams.classNLDeriv, mergeOutput);

          % Accumulate across batches.
    end

    MergeDeltaLeft = MergeDeltaLeft .* leftMask;
    MergeDeltaRight = MergeDeltaRight .* rightMask;

    [ localWordFeatureGradients, ...
      localConnectionMatrixGradients, ...
      localCompositionMatrixGradients, ...
      localEmbeddingTransformMatrixGradients ] = ...
       leftBatch.getGradient(MergeDeltaLeft, wordFeatures, embeddingTransformMatrix, ...
                            connectionMatrix, compositionMatrix, hyperParams);

    [ rightWordGradients, ...
      rightConnectionMatrixGradients, ...
      rightCompositionMatrixGradients, ...
      rightEmbeddingTransformMatrixGradients ] = ...
       rightBatch.getGradient(MergeDeltaRight, wordFeatures, embeddingTransformMatrix, ...
                            connectionMatrix, compositionMatrix, hyperParams);

    if hyperParams.trainWords
        localWordFeatureGradients = localWordFeatureGradients ...
            + rightWordGradients;
    end
    localConnectionMatrixGradients = localConnectionMatrixGradients...
        + rightConnectionMatrixGradients;
    localCompositionMatrixGradients = localCompositionMatrixGradients...
        + rightCompositionMatrixGradients;
    localEmbeddingTransformMatrixGradients = localEmbeddingTransformMatrixGradients...
        + rightEmbeddingTransformMatrixGradients;
    
    % Pack up gradients
    if hyperParams.fastEmbed
        grad = param2stack(localMergeMatricesGradients, ...
            localMergeMatrixGradients, ...
            localSoftmaxGradient, ...
            [], localConnectionMatrixGradients, ...
            localCompositionMatrixGradients, ...
            localExtraMatrixGradients, ...
            localEmbeddingTransformMatrixGradients);
    else
        grad = param2stack(localMergeMatricesGradients, ...
            localMergeMatrixGradients, localSoftmaxGradient, ...
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
            % assert(sum(isnan(embGrad(:, wordInd))) == 0, 'NaNs in computed embedding gradient.');
            % assert(sum(isinf(embGrad(:, wordInd))) == 0, 'Infs in computed embedding gradient.');
        end
    else
        embGrad = [];
    end

    if sum(isnan(grad)) > 0
        [ mergeMatrices, mergeMatrix, ...
            softmaxMatrix, trainedWordFeatures, compositionMatrices,...
            compositionMatrix, classifierExtraMatrix, ...
            embeddingTransformMatrix ] ...
            = stack2param(grad, decoder);
        mergeMatrices, mergeMatrix, ...
            softmaxMatrix, trainedWordFeatures, compositionMatrices, ...
            compositionMatrix, classifierExtraMatrix, ...
            embeddingTransformMatrix
        assert(false, 'NANs in computed gradient.');
    end

    assert(sum(isinf(grad)) == 0, 'Infs in computed gradient.'); 
end

end
