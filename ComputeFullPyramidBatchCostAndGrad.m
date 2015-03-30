% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, embGrad, acc, confusion ] = ComputeFullPyramidBatchCostAndGrad(theta, decoder, data, separateWordFeatures, hyperParams, computeGrad)
% Compute cost, gradient, accuracy, and confusions over a batch of examples for some parameters.
% This is a well-behaved costGradFn, and can be @-passed to optimizers, including minFunc and TrainSGD.

% TODO: Make compatible with sequences.

B = length(data);  % Batch size.
D = hyperParams.dim;  % Sentence embedding dimension.

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
leftPyramidBatch = PyramidBatch.makePyramidBatch([data(:).left], wordFeatures, hyperParams);
rightPyramidBatch = PyramidBatch.makePyramidBatch([data(:).right], wordFeatures, hyperParams);
[ leftFeatures, leftConnectionCosts ] = leftPyramidBatch.runForward(connectionMatrix, compositionMatrix,hyperParams);
[ rightFeatures, rightConnectionCosts ] = rightPyramidBatch.runForward(connectionMatrix, compositionMatrix, hyperParams);

% Set up and run top dropout.
if nargout > 1 || hyperParams.minFunc
    bottomDropout = hyperParams.bottomDropout;
    topDropout = hyperParams.topDropout;
else
    bottomDropout = 1;
    topDropout = 1;
end
[ leftFeatures, leftMask ] = Dropout(leftFeatures, topDropout);
[ rightFeatures, rightMask ] = Dropout(rightFeatures, topDropout);

% Compute classification tensor layer (or plain RNN layer).
if hyperParams.useThirdOrderMerge
    [ mergeOutput, tensorInnerOutput ] = ComputeTensorLayer(leftFeatures, ...
        rightFeatures, mergeMatrices, mergeMatrix, hyperParams.classNL);
else
    [ mergeOutput, innerOutput ] = ComputeRNNLayer(leftFeatures, rightFeatures, ...
        mergeMatrix, hyperParams.classNL);
end

% TODO: Add post-merge layers back in

[ relationProbs, topCosts ] = ComputeSoftmaxLayer(mergeOutput, softmaxMatrix, 1:size(softmaxMatrix, 1), [data(:).relation]);

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
confusion = zeros(hyperParams.numRelations);
for b = 1:B
    localCorrect = preds(b) == data(b).relation(find(data(b).relation > 0));
    accumulatedSuccess = accumulatedSuccess + localCorrect;

    if (~localCorrect) && (nargout > 2) && hyperParams.showExamples
        Log(hyperParams.examplelog, ['for: ', data(b).left.getText(), ' ', data(b).right.getText(), ...
              ' hyp:  ', num2str(preds(b)), ' w/ p=', num2str(relationProbs(preds(b), b))]);
    end

    if nargout > 4
        confusion(preds(b), data(b).relation(find(data(b).relation > 0))) = ...
          confusion(preds(b), data(b).relation(find(data(b).relation > 0))) + 1;
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
    % TODO: Add back extra post-merge layer support here.
    [ localSoftmaxGradient, softmaxDelta ] = ...
        ComputeSoftmaxClassificationGradients(...
          softmaxMatrix, relationProbs, [data(:).relation], mergeOutput);
    localSoftmaxGradient = sum(localSoftmaxGradient, 3);

    if hyperParams.useThirdOrderMerge
        % Compute gradients for the merge tensor layer
        [ localMergeMatricesGradients, ...
            localMergeMatrixGradients, ...
            MergeDeltaLeft, ...
            MergeDeltaRight ] = ...
          ComputeTensorLayerGradients(leftFeatures, rightFeatures, ...
              mergeMatrices, mergeMatrix, ...
              softmaxDelta, hyperParams.classNLDeriv, tensorInnerOutput);
          localMergeMatricesGradients = sum(localMergeMatricesGradients, 4);
          localMergeMatrixGradients = sum(localMergeMatrixGradients, 3);
    else
         % Compute gradients for the merge NN layer
         localMergeMatricesGradients = [];  
         [localMergeMatrixGradients, ...
            MergeDeltaLeft, ...
            MergeDeltaRight] = ...
          ComputeRNNLayerGradients(leftFeatures, rightFeatures, ...
              mergeMatrix, ...
              softmaxDelta, hyperParams.classNLDeriv, innerOutput);

          % Accumulate across batches.
          localMergeMatrixGradients = sum(localMergeMatrixGradients, 3);
    end

    MergeDeltaLeft = MergeDeltaLeft .* leftMask;
    MergeDeltaRight = MergeDeltaRight .* rightMask;

    [ localWordFeatureGradients, ...
      localConnectionMatrixGradients, ...
      localCompositionMatrixGradients, ...
      localEmbeddingTransformMatrixGradients ] = ...
       leftPyramidBatch.getGradient(MergeDeltaLeft, [], wordFeatures, ...
                            connectionMatrix, compositionMatrix, ...
                            embeddingTransformMatrix, hyperParams.compNLDeriv, hyperParams);

    [ rightWordGradients, ...
      rightConnectionMatrixGradients, ...
      rightCompositionMatrixGradients, ...
      rightEmbeddingTransformMatrixGradients ] = ...
       rightPyramidBatch.getGradient(MergeDeltaRight, [], wordFeatures, ...
                            connectionMatrix, compositionMatrix, ...
                            embeddingTransformMatrix, hyperParams.compNLDeriv, hyperParams);

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
          [], ...
          localEmbeddingTransformMatrixGradients);
      embGrad = localWordFeatureGradients;
    else
      [grad, dec] = param2stack(localMergeMatricesGradients, ...
          localMergeMatrixGradients, localSoftmaxGradient, ...
          localWordFeatureGradients, localConnectionMatrixGradients, ...
          localCompositionMatrixGradients, ...
          [], ...
          localEmbeddingTransformMatrixGradients); 
      embGrad = [];
    end

    % Clip the gradient.
    % TODO: Must separate gradients back out into batches to make this doable here?
    % if hyperParams.clipGradients
    %     gradNorm = norm(grad);
    %     if gradNorm > hyperParams.maxGradNorm
    %         grad = grad ./ gradNorm;
    %     end
    % end

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


end
