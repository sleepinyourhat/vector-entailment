% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, embGrad, pred ] = ComputeSentenceClassificationExampleCostAndGrad(theta, thetaDecoder, dataPoint, separateWordFeatures, hyperParams, computeGradient)
% Compute cost, gradient, and predicted label for one example.

grad = [];
embGrad = [];

assert(~hyperParams.useLattices, 'Unbatched computation is not available for the LatticeNN. Use batching.');
assert(~hyperParams.gpu, 'Use batching for GPU computation.');

% Unpack theta
[ ~, ~, ...
    softmaxMatrix, trainedWordFeatures, compositionMatrices,...
    compositionMatrix, ~, classifierExtraMatrix, ...
    embeddingTransformMatrix ] ...
    = stack2param(theta, thetaDecoder);

if hyperParams.trainWords && ~hyperParams.largeVocabMode
    wordFeatures = trainedWordFeatures;
else
    wordFeatures = separateWordFeatures;
end

DIM = hyperParams.dim;
EMBDIM = hyperParams.embeddingDim;

if isfield(hyperParams, 'smallVecs') && hyperParams.smallVecs
    sigma = 0.25;
else
    sigma = 1;
end

wordFeatures(:, hyperParams.randomEmbeddingIndices) = ...
    fNormrnd(0, sigma, [EMBDIM, length(hyperParams.randomEmbeddingIndices)], ...
             hyperParams.gpu, hyperParams.gpu && hyperParams.largeVocabMode);
wordFeatures(1, hyperParams.randomEmbeddingIndices) = 1;

% Set the number of mposition functions
if hyperParams.useSumming
    NUMCOMP = 0;
elseif ~hyperParams.untied
    NUMCOMP = 1;
else
    NUMCOMP = 3;
end

NUMTRANS = size(embeddingTransformMatrix, 3);

% Run the trees/sequences forward
dataPoint.sentence.updateFeatures(wordFeatures, compositionMatrices, ...
        compositionMatrix, embeddingTransformMatrix, hyperParams.compNL, computeGradient, hyperParams);

[ features, mask ] = Dropout(dataPoint.sentence.getFeatures(), hyperParams.topDropout, computeGradient, hyperParams.gpu);
       
% Run layers forward
extraInputs = zeros(hyperParams.penultDim, 1, hyperParams.topDepth);
extraInnerOutputs = zeros(hyperParams.penultDim, 1, hyperParams.topDepth - 1);
extraInputs(:, 1, 1) = features;
for layer = 1:(hyperParams.topDepth - 1) 
    extraInnerOutputs(:, 1, layer) = classifierExtraMatrix(:, :, layer) * [1; extraInputs(:, 1, layer)];
    extraInputs(:, 1, layer + 1) = hyperParams.classNL(extraInnerOutputs(:, 1, layer));
end

if ~isempty(hyperParams.labelCostMultipliers)
    multiplier = hyperParams.labelCostMultipliers(dataPoint.label(1));
    [ labelProbs, cost ] = ComputeSoftmaxLayer( ...
                        extraInputs(:,hyperParams.topDepth), softmaxMatrix, hyperParams, dataPoint.label', multiplier);
else
    [ labelProbs, cost ] = ComputeSoftmaxLayer( ...
                      extraInputs(:,hyperParams.topDepth), softmaxMatrix, hyperParams, dataPoint.label');  
end


% Produce gradient
if nargout > 1 && (nargin < 6 || computeGradient)
    if ~isempty(hyperParams.labelCostMultipliers)
        [ localSoftmaxGradient, softmaxDelta ] = ...
            ComputeSoftmaxClassificationGradients( ...
              softmaxMatrix, labelProbs, dataPoint.label', ...
              extraInputs(:,hyperParams.topDepth), hyperParams, multiplier);
    else
        [ localSoftmaxGradient, softmaxDelta ] = ...
            ComputeSoftmaxClassificationGradients( ...
              softmaxMatrix, labelProbs, dataPoint.label', ...
              extraInputs(:,hyperParams.topDepth), hyperParams);
    end

    % Compute gradients for extra top layers
    [ localExtraMatrixGradients, extraDelta ] = ...
          ComputeExtraClassifierGradients(classifierExtraMatrix,...
            softmaxDelta, extraInputs, hyperParams.classNLDeriv);

    extraDelta = extraDelta .* mask;

    [ localWordFeatureGradients, ...
      localCompositionMatricesGradients, ...
      localCompositionMatrixGradients, ...
      localEmbeddingTransformMatrixGradients ] = ...
       dataPoint.sentence.getGradient(extraDelta, [], wordFeatures, ...
                            compositionMatrices, compositionMatrix, ...
                            embeddingTransformMatrix, ...
                            hyperParams.compNLDeriv, hyperParams);
    
    % Pack up gradients
    if hyperParams.largeVocabMode
      grad = param2stack([], ...
          [], ...
          localSoftmaxGradient, ...
          [], localCompositionMatricesGradients, ...
          localCompositionMatrixGradients, ...
          [], ...
          localExtraMatrixGradients, ...
          localEmbeddingTransformMatrixGradients);
      embGrad = localWordFeatureGradients;
    else
      [ grad, dec ] = param2stack([], ...
          [], ...
          localSoftmaxGradient, ...
          localWordFeatureGradients, localCompositionMatricesGradients, ...
          localCompositionMatrixGradients, ...
          [], ...
          localExtraMatrixGradients, ...
          localEmbeddingTransformMatrixGradients); 
      embGrad = [];
    end

    % Clip the gradient.
    if hyperParams.clipGradients
        gradNorm = norm(grad);
        if gradNorm > hyperParams.maxGradNorm
            grad = grad ./ gradNorm;
        end
    end
end

% Compute prediction. Note: This will be in integer, indexing into whichever class set was used
% for this example.
if nargout > 3
    [ ~, pred ] = max(labelProbs);
end

end

