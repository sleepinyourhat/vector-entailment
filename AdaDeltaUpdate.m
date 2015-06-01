function modelState = AdaDeltaUpdate(modelState, options, hyperParams, grad, embGrad)
% TODO: This is pretty slow, and could benefit from smart GPUification.

if modelState.step == 0
    modelState.sumSqGrad = fZeros(size(modelState.theta), hyperParams.gpu);
    modelState.sumSqDelta = fZeros(size(modelState.theta), hyperParams.gpu);
    if length(embGrad) > 0
        % Set up a separate SumSqGrad tracker for the embeddings.
        modelState.sumSqEmbGrad = fZeros(size(modelState.separateWordFeatures), hyperParams.gpu && ~hyperParams.largeVocabMode);
        modelState.sumSqEmbDelta = fZeros(size(modelState.separateWordFeatures), hyperParams.gpu && ~hyperParams.largeVocabMode);
    end
end

% Do an AdaDelta-scaled parameter update
modelState.sumSqGrad = modelState.sumSqGrad * options.adaDeltaRho + (grad .^ 2) * (1 - options.adaDeltaRho);

rmsLastDelta = sqrt(modelState.sumSqDelta + options.adaDeltaEps);
rmsGrad = sqrt(modelState.sumSqGrad + options.adaDeltaEps);

delta = -1 * (rmsLastDelta ./ rmsGrad) .* grad;
modelState.sumSqDelta = modelState.sumSqDelta * options.adaDeltaRho + (delta .^ 2) * (1 - options.adaDeltaRho);

modelState.theta = modelState.theta + delta;

% Do an AdaDelta-scaled parameter update to the separate word features
if length(embGrad) > 0
    modelState.sumSqEmbGrad = modelState.sumSqEmbGrad * options.adaDeltaRho + (embGrad .^ 2) * (1 - options.adaDeltaRho);

    rmsLastEmbDelta = sqrt(modelState.sumSqEmbDelta + options.adaDeltaEps);
    rmsEmbGrad = sqrt(modelState.sumSqEmbGrad + options.adaDeltaEps);

    embDelta = -1 * (rmsLastEmbDelta ./ rmsEmbGrad) .* embGrad;
    modelState.sumSqEmbDelta = modelState.sumSqEmbDelta * options.adaDeltaRho + (embDelta .^ 2) * (1 - options.adaDeltaRho);

    modelState.separateWordFeatures = modelState.separateWordFeatures + embDelta;
end

end
