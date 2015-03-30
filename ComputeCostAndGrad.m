% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, embGrad, pred ] = ComputeCostAndGrad(theta, thetaDecoder, dataPoint, separateWordFeatures, hyperParams, computeGradient)
% Compute cost, gradient, and predicted label for one example.

grad = [];
embGrad = [];

% Unpack theta
[ mergeMatrices, mergeMatrix ...
    softmaxMatrix, trainedWordFeatures, compositionMatrices,...
    compositionMatrix, classifierExtraMatrix, ...
    embeddingTransformMatrix ] ...
    = stack2param(theta, thetaDecoder);

if hyperParams.trainWords && ~hyperParams.fastEmbed
    wordFeatures = trainedWordFeatures;
else
    wordFeatures = separateWordFeatures;
end

DIM = hyperParams.dim;
EMBDIM = hyperParams.embeddingDim;

% Set the number of composition functions
if hyperParams.useSumming
    NUMCOMP = 0;
elseif ~hyperParams.untied
    NUMCOMP = 1;
else
    NUMCOMP = 3;
end

NUMTRANS = size(embeddingTransformMatrix, 3);

left = dataPoint.left;
right = dataPoint.right;
trueRelation = dataPoint.relation;

relationRange = ComputeRelationRange(hyperParams, trueRelation);

if nargout > 1 || hyperParams.minFunc
  bottomDropout = hyperParams.bottomDropout;
  topDropout = hyperParams.topDropout;
else
  bottomDropout = 1;
  topDropout = 1;
end

% Run the trees/sequences forward
left.updateFeatures(wordFeatures, compositionMatrices, ...
        compositionMatrix, embeddingTransformMatrix, hyperParams.compNL, bottomDropout);
right.updateFeatures(wordFeatures, compositionMatrices, ...
        compositionMatrix, embeddingTransformMatrix, hyperParams.compNL, bottomDropout);

leftFeatures = left.getFeatures();
rightFeatures = right.getFeatures();

[leftFeatures, leftMask] = Dropout(leftFeatures, topDropout);
[rightFeatures, rightMask] = Dropout(rightFeatures, topDropout);

% Compute classification tensor layer (or plain RNN layer)
if hyperParams.useThirdOrderMerge
    [classTensorOutput, tensorInnerOutput] = ComputeTensorLayer(leftFeatures, ...
        rightFeatures, mergeMatrices, mergeMatrix, hyperParams.classNL);
else
    [classTensorOutput, innerOutput] = ComputeRNNLayer(leftFeatures, rightFeatures, ...
        mergeMatrix, hyperParams.classNL);
end
       
% Run layers forward
extraInputs = zeros(hyperParams.penultDim, hyperParams.topDepth);
extraInnerOutputs = zeros(hyperParams.penultDim, hyperParams.topDepth - 1);
extraInputs(:,1) = classTensorOutput;
for layer = 1:(hyperParams.topDepth - 1) 
    extraInnerOutputs(:,layer) = classifierExtraMatrix(:,:,layer) * [1, extraInputs(:,layer)];
    extraInputs(:,layer + 1) = hyperParams.classNL(extraInnerOutputs(:,layer));
end

[ relationProbs, cost ] = ComputeSoftmaxLayer( ...
                    extraInputs(:,hyperParams.topDepth), softmaxMatrix, relationRange, max(trueRelation));

% Produce gradient
if nargout > 1 && (nargin < 6 || computeGradient)
    
    [ localSoftmaxGradient, softmaxDelta ] = ...
        ComputeSoftmaxClassificationGradients( ...
          softmaxMatrix, relationProbs, max(trueRelation), ...
          extraInputs(:,hyperParams.topDepth), relationRange);
    
    % Compute gradients for extra top layers
    [ localExtraMatrixGradients, extraDelta ] = ...
          ComputeExtraClassifierGradients(classifierExtraMatrix,...
          softmaxDelta, extraInputs, extraInnerOutputs, hyperParams.classNLDeriv);

    if hyperParams.useThirdOrderMerge
        % Compute gradients for classification tensor layer
        [ localMergeMatricesGradients, localMergeMatrixGradients, ...
            MergeDeltaLeft, MergeDeltaRight ] = ...
            ComputeTensorLayerGradients(leftFeatures, rightFeatures, ...
              mergeMatrices, mergeMatrix, ...
              extraDelta, hyperParams.classNLDeriv, tensorInnerOutput);
    else
         % Compute gradients for classification first layer
         localMergeMatricesGradients = zeros(0, 0, 0);  
         [ localMergeMatrixGradients, MergeDeltaLeft, ...
            MergeDeltaRight ] = ...
          ComputeRNNLayerGradients(leftFeatures, rightFeatures, ...
              mergeMatrix, ...
              extraDelta, hyperParams.classNLDeriv, innerOutput);
    end

    MergeDeltaLeft = MergeDeltaLeft .* leftMask;
    MergeDeltaRight = MergeDeltaRight .* rightMask;

    [ localWordFeatureGradients, ...
      localCompositionMatricesGradients, ...
      localCompositionMatrixGradients, ...
      localEmbeddingTransformMatrixGradients ] = ...
       left.getGradient(MergeDeltaLeft, [], wordFeatures, ...
                            compositionMatrices, compositionMatrix, ...
                            embeddingTransformMatrix, ...
                            hyperParams.compNLDeriv, hyperParams);

    [ rightWordGradients, ...
      rightCompositionMatricesGradients, ...
      rightCompositionMatrixGradients, ...
      rightEmbeddingTransformMatrixGradients ] = ...
       right.getGradient(MergeDeltaRight, [], wordFeatures, ...
                            compositionMatrices, compositionMatrix, ...
                            embeddingTransformMatrix, ...
                            hyperParams.compNLDeriv, hyperParams);
    if hyperParams.trainWords
      localWordFeatureGradients = localWordFeatureGradients ...
          + rightWordGradients;
    end
    localCompositionMatricesGradients = localCompositionMatricesGradients...
        + rightCompositionMatricesGradients;
    localCompositionMatrixGradients = localCompositionMatrixGradients...
        + rightCompositionMatrixGradients;
    localEmbeddingTransformMatrixGradients = localEmbeddingTransformMatrixGradients...
        + rightEmbeddingTransformMatrixGradients;
    
    % Pack up gradients
    if hyperParams.fastEmbed
      grad = param2stack(localMergeMatricesGradients, ...
          localMergeMatrixGradients, ...
          ocalSoftmaxGradient, ...
          [], localCompositionMatricesGradients, ...
          localCompositionMatrixGradients, ...
          localExtraMatrixGradients, ...
          localEmbeddingTransformMatrixGradients);
      embGrad = localWordFeatureGradients;
    else
      [ grad, dec ] = param2stack(localMergeMatricesGradients, ...
          localMergeMatrixGradients, ...
          localSoftmaxGradient, ...
          localWordFeatureGradients, localCompositionMatricesGradients, ...
          localCompositionMatrixGradients, ...
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

% This doesn't appear to save any memory, but may be woth revisiting for large datasets.
% if hyperParams.clearActivations
%     left.clearActivations()
%     right.clearActivations()
% end

% Compute prediction. Note: This will be in integer, indexing into whichever class set was used
% for this example.
if nargout > 3
    [ ~, pred ] = max(relationProbs);
end

end

