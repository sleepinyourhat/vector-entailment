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
    compositionMatrix, scoringVector, classifierExtraMatrix, embeddingTransformMatrix] ...
    = stack2param(theta, decoder);
if hyperParams.trainWords && ~hyperParams.largeVocabMode
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
    leftBatch.runForward(embeddingTransformMatrix, connectionMatrix, scoringVector, compositionMatrix, hyperParams, computeGrad);
[ rightFeatures, rightConnectionCosts, rightConnectionAcc ] = ...
    rightBatch.runForward(embeddingTransformMatrix, connectionMatrix, scoringVector, compositionMatrix, hyperParams, computeGrad);

% TODO: Weighted average.
connectionAcc = [leftConnectionAcc rightConnectionAcc];

% Set up and run top dropout.
[ leftFeatures, leftMask ] = Dropout(leftFeatures, hyperParams.topDropout, computeGrad, hyperParams.gpu);
[ rightFeatures, rightMask ] = Dropout(rightFeatures, hyperParams.topDropout, computeGrad, hyperParams.gpu);

% Compute classification tensor layer (or plain RNN layer).
if hyperParams.useThirdOrderMerge
    mergeOutput = ComputeTensorLayer(leftFeatures, ...
        rightFeatures, mergeMatrices, mergeMatrix, hyperParams.classNL);
else
    mergeOutput = ComputeRNNLayer(leftFeatures, rightFeatures, ...
        mergeMatrix, hyperParams.classNL);
end

% Post-merge layers
extraClassifierLayerInputs = fZeros([hyperParams.penultDim, B, hyperParams.topDepth], hyperParams.gpu);
extraClassifierLayerInnerOutputs = fZeros([hyperParams.penultDim, B, hyperParams.topDepth - 1], hyperParams.gpu);
extraClassifierLayerInputs(:, :, 1) = permute(mergeOutput, [1, 3, 2]);
for layer = 1:(hyperParams.topDepth - 1) 
    extraClassifierLayerInnerOutputs(:, :, layer) = classifierExtraMatrix(:, :, layer) * ...
        [fOnes([1, B], hyperParams.gpu); extraClassifierLayerInputs(:, :, layer)];
    extraClassifierLayerInputs(:, :, layer + 1) = hyperParams.classNL(extraClassifierLayerInnerOutputs(:, :, layer));
end

if ~isempty(hyperParams.labelCostMultipliers)
    multipliers = hyperParams.labelCostMultipliers([data(:).label])';
    multipliers = multipliers(:, 1);
    [ labelProbs, topCosts ] = ComputeSoftmaxLayer(extraClassifierLayerInputs(:, :, hyperParams.topDepth), ...
                  softmaxMatrix, hyperParams, [data(:).label]', multipliers);
elseif hyperParams.sentimentBigramMode
    [ labelProbs, topCosts ] = ComputeSoftmaxLayer(extraClassifierLayerInputs(:, :, hyperParams.topDepth), ...
                  softmaxMatrix, hyperParams, [], [], [], [data(:).score_dist]);
else
    [ labelProbs, topCosts ] = ComputeSoftmaxLayer(extraClassifierLayerInputs(:, :, hyperParams.topDepth), ...
                  softmaxMatrix, hyperParams, [data(:).label]');
end

% Sum the log losses from the three sources over all of the batch elements and normalize.
% TODO: Is it worth scaling the two different types of cost?
normalizedCost = sum([topCosts; leftConnectionCosts; rightConnectionCosts]) / length(data);

% Apply regularization to the cost (does not include largeVocabMode embeddings).
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
[ ~, preds ] = max(labelProbs);
if ~hyperParams.sentimentBigramMode
    confusion = zeros(hyperParams.numLabels(data(1).label(2)));
    for b = 1:B
            localCorrect = preds(b) == data(b).label(1);
            accumulatedSuccess = accumulatedSuccess + localCorrect;

        if (~localCorrect) && (nargout > 2) && hyperParams.showExamples
            Log(hyperParams.examplelog, ...
                ['hyp:' num2str(preds(b)), ' ---- true:', num2str(data(b).label(1)), '---- ',...
                 data(b).left.getText(), ' ---- ', data(b).right.getText()]);
        end

        if nargout > 5
            confusion(preds(b), data(b).label(1)) = ...
              confusion(preds(b), data(b).label(1)) + 1;
        end
    end
    acc = (accumulatedSuccess / B);
else
    if hyperParams.showExamples && (nargout > 2)
        for b = 1:B
            Log(hyperParams.examplelog, ...
                ['hyp:' num2str(labelProbs(:,b)'), ' ---- true: ', num2str(data(b).score_dist'), ' ---- ',...
                 data(b).left.getText(), ' ---- ', data(b).right.getText()]);  
        end 
    end

    confusion = 0;
    acc = normalizedCost;
end


% Compute the gradients.
if computeGrad
    if ~isempty(hyperParams.labelCostMultipliers)
        [ localSoftmaxGradient, softmaxDelta ] = ...
        ComputeSoftmaxClassificationGradients(...
          softmaxMatrix, labelProbs, [data(:).label]', extraClassifierLayerInputs(:, :, hyperParams.topDepth), hyperParams, ...
           multipliers);
    elseif hyperParams.sentimentBigramMode
        [ localSoftmaxGradient, softmaxDelta ] = ...
        ComputeSoftmaxClassificationGradients(...
          softmaxMatrix, labelProbs, [], extraClassifierLayerInputs(:, :, hyperParams.topDepth), hyperParams, [], [data(:).score_dist]);
    else
        [ localSoftmaxGradient, softmaxDelta ] = ...
        ComputeSoftmaxClassificationGradients(...
          softmaxMatrix, labelProbs, [data(:).label]', extraClassifierLayerInputs(:, :, hyperParams.topDepth), hyperParams);
    end
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
      localScoringVectorGradients, ...
      localCompositionMatrixGradients, ...
      localEmbeddingTransformMatrixGradients ] = ...
       leftBatch.getGradient(MergeDeltaLeft, wordFeatures, embeddingTransformMatrix, ...
                            connectionMatrix, scoringVector, compositionMatrix, hyperParams);

    [ rightWordGradients, ...
      rightConnectionMatrixGradients, ...
      rightScoringVectorGradients, ...
      rightCompositionMatrixGradients, ...
      rightEmbeddingTransformMatrixGradients ] = ...
       rightBatch.getGradient(MergeDeltaRight, wordFeatures, embeddingTransformMatrix, ...
                            connectionMatrix, scoringVector, compositionMatrix, hyperParams);

    if hyperParams.trainWords
        localWordFeatureGradients = localWordFeatureGradients ...
            + rightWordGradients;
    end
    localConnectionMatrixGradients = localConnectionMatrixGradients...
        + rightConnectionMatrixGradients;
    localScoringVectorGradients = localScoringVectorGradients...
        + rightScoringVectorGradients;
    localCompositionMatrixGradients = localCompositionMatrixGradients...
        + rightCompositionMatrixGradients;
    localEmbeddingTransformMatrixGradients = localEmbeddingTransformMatrixGradients...
        + rightEmbeddingTransformMatrixGradients;
    
    % Pack up gradients
    if hyperParams.largeVocabMode
        grad = param2stack(localMergeMatricesGradients, ...
            localMergeMatrixGradients, ...
            localSoftmaxGradient, ...
            [], localConnectionMatrixGradients, ...
            localCompositionMatrixGradients, ...
            localScoringVectorGradients, ...
            localExtraMatrixGradients, ...
            localEmbeddingTransformMatrixGradients);
    else
        grad = param2stack(localMergeMatricesGradients, ...
            localMergeMatrixGradients, localSoftmaxGradient, ...
            localWordFeatureGradients, localConnectionMatrixGradients, ...
            localCompositionMatrixGradients, ...
            localScoringVectorGradients, ...
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

    if hyperParams.largeVocabMode
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

    if hyperParams.clipGradients
        gradNorm = norm(grad);
        if gradNorm > hyperParams.maxGradNorm
            grad = grad .* (hyperParams.maxGradNorm ./ gradNorm);
        end
    end

    if sum(isnan(grad)) > 0
        [ mergeMatrices, mergeMatrix, ...
            softmaxMatrix, trainedWordFeatures, compositionMatrices,...
            compositionMatrix, scoringVector, classifierExtraMatrix, ...
            embeddingTransformMatrix ] ...
            = stack2param(grad, decoder);
        mergeMatrices, mergeMatrix, ...
            softmaxMatrix, trainedWordFeatures, compositionMatrices, ...
            compositionMatrix, scoringVector, classifierExtraMatrix, ...
            embeddingTransformMatrix
        assert(false, 'NANs in computed gradient.');
    end

    assert(sum(isinf(grad)) == 0, 'Infs in computed gradient.'); 
end

end
