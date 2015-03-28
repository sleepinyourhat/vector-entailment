% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, embGrad, acc, confusion ] = ComputeFullPyramidBatchCostAndGrad(theta, decoder, data, separateWordFeatures, hyperParams, computeGrad)
% Compute cost, gradient, accuracy, and confusions over a set of examples for some parameters.

B = length(data);
D = hyperParams.dim;

if nargout > 4
    confusions = zeros(N, 2);
end

% Unpack theta
[~, classifierMatrix, ...
    softmaxMatrix, trainedWordFeatures, connectionMatrix, ...
    compositionMatrix, classifierExtraMatrix, embeddingTransformMatrix] ...
    = stack2param(theta, decoder);

if hyperParams.trainWords && ~hyperParams.fastEmbed
    wordFeatures = trainedWordFeatures;
else
    wordFeatures = separateWordFeatures;
end

leftPyramidBatch = PyramidBatch.makePyramidBatch([data(:).left], wordFeatures, hyperParams);
rightPyramidBatch = PyramidBatch.makePyramidBatch([data(:).right], wordFeatures, hyperParams);

[ leftFeatures ] = leftPyramidBatch.runForward(connectionMatrix, compositionMatrix,hyperParams);
[ rightFeatures ] = rightPyramidBatch.runForward(connectionMatrix, compositionMatrix, hyperParams);

% Set up and run top dropout
if nargout > 1 || hyperParams.minFunc
  bottomDropout = hyperParams.bottomDropout;
  topDropout = hyperParams.topDropout;
else
  bottomDropout = 1;
  topDropout = 1;
end
[ leftFeatures, leftMask ] = Dropout(leftFeatures, topDropout);
[ rightFeatures, rightMask ] = Dropout(rightFeatures, topDropout);

% Compute classification tensor layer (or plain RNN layer)
if hyperParams.useThirdOrderComparison
    [ mergeOutput, tensorInnerOutput ] = ComputeTensorLayer(leftFeatures, ...
        rightFeatures, classifierMatrices, classifierMatrix, hyperParams.classNL);
else
    [ mergeOutput, innerOutput ] = ComputeRNNLayer(leftFeatures, rightFeatures, ...
        classifierMatrix, hyperParams.classNL);
end

% TODO: Add post-merge layers back in

relationProbs = ComputeSoftmaxProbabilities(mergeOutput, softmaxMatrix);

% Compute cost
cost = BatchObjective([data(:).relation], relationProbs, hyperParams);

%%%% Gradient %%%%

% Compute mean cost
normalizedCost = cost / length(data);

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
    % TODO: Add back extra post-merge layer support here.
    [localSoftmaxGradient, softmaxDelta] = ...
        ComputeBatchSoftmaxClassificationGradient(hyperParams, softmaxMatrix, ...
                               relationProbs, [data(:).relation], mergeOutput);
    localSoftmaxGradient = sum(localSoftmaxGradient, 3);
    
    if hyperParams.useThirdOrderComparison
        % Compute gradients for classification tensor layer
        [localClassificationMatricesGradients, ...
            localClassificationMatrixGradients, ...
            classifierDeltaLeft, ...
            classifierDeltaRight] = ...
          ComputeTensorLayerGradients(leftFeatures, rightFeatures, ...
              classifierMatrices, classifierMatrix, ...
              softmaxDelta, hyperParams.classNLDeriv, tensorInnerOutput);
    else
         % Compute gradients for classification first layer
         localClassificationMatricesGradients = zeros(0, 0, 0);  
         [localClassificationMatrixGradients, ...
            classifierDeltaLeft, ...
            classifierDeltaRight] = ...
          ComputeRNNLayerGradients(leftFeatures, rightFeatures, ...
              classifierMatrix, ...
              softmaxDelta, hyperParams.classNLDeriv, innerOutput);
    end

    localClassificationMatrixGradients

    classifierDeltaLeft = classifierDeltaLeft .* leftMask;
    classifierDeltaRight = classifierDeltaRight .* rightMask;

    [ localWordFeatureGradients, ...
      localConnectionMatrixGradients, ...
      localCompositionMatrixGradients, ...
      localEmbeddingTransformMatrixGradients ] = ...
       leftPyramidBatch.getGradient(classifierDeltaLeft, [], wordFeatures, ...
                            connectionMatrix, compositionMatrix, ...
                            embeddingTransformMatrix, ...
                            hyperParams.compNLDeriv, hyperParams);

    [ rightWordGradients, ...
      rightConnectionMatrixGradients, ...
      rightCompositionMatrixGradients, ...
      rightEmbeddingTransformMatrixGradients ] = ...
       rightPyramidBatch.getGradient(classifierDeltaRight, [], wordFeatures, ...
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
      grad = param2stack(localClassificationMatricesGradients, ...
          localClassificationMatrixGradients, ...
          localSoftmaxGradient, ...
          [], localConnectionMatrixGradients, ...
          localCompositionMatrixGradients, ...
          [], ...
          localEmbeddingTransformMatrixGradients);
      embGrad = localWordFeatureGradients;
    else
      [grad, dec] = param2stack(localClassificationMatricesGradients, ...
          localClassificationMatrixGradients, localSoftmaxGradient, ...
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

if nargout > 3
    acc = (accumulatedSuccess / N);
end

end
