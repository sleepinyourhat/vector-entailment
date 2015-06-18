function modelState = AdaDeltaUpdate(modelState, options, grad, embGrad)

if modelState.step == 0
    modelState.sumSqGrad = zeros(size(modelState.theta));
    modelState.sumSqDelta = zeros(size(modelState.theta));
    if length(embGrad) > 0
        % Set up a separate SubSqGrad tracker for the embeddings.
        modelState.sumSqEmbGrad = zeros(size(modelState.separateWordFeatures));
        modelState.sumSqEmbDelta = zeros(size(modelState.separateWordFeatures));
    end
end

% Do an AdaGrad-scaled parameter update
modelState.sumSqGrad = modelState.sumSqGrad * options.adaDeltaRho + (grad.^2) * (1 - options.adaDeltaRho);

rmsLastDelta = sqrt(modelState.sumSqDelta + options.adaDeltaEps);
rmsGrad = sqrt(modelState.sumSqGrad + options.adaDeltaEps);

delta = -1 * (rmsLastDelta ./ rmsGrad) .* grad;
modelState.sumSqDelta = modelState.sumSqDelta * options.adaDeltaRho + (delta.^2) * (1 - options.adaDeltaRho);

modelState.theta = modelState.theta + delta;

assert(sum(isnan(modelState.theta)) == 0, 'NaNs in theta.');
assert(sum(isinf(modelState.theta)) == 0, 'Infs in theta.');

% Do an AdaGrad-scaled parameter update to the separate word features
if length(embGrad) > 0
    modelState.sumSqEmbGrad = modelState.sumSqEmbGrad * options.adaDeltaRho + (embGrad.^2) * (1 - options.adaDeltaRho);

    rmsLastEmbDelta = sqrt(modelState.sumSqEmbDelta + options.adaDeltaEps);
    rmsEmbGrad = sqrt(modelState.sumSqEmbGrad + options.adaDeltaEps);

    embDelta = -1 * (rmsLastEmbDelta ./ rmsEmbGrad) .* embGrad;
    modelState.sumSqEmbDelta = modelState.sumSqEmbDelta * options.adaDeltaRho + (embDelta.^2) * (1 - options.adaDeltaRho);

    modelState.separateWordFeatures = modelState.separateWordFeatures + embDelta;
end

end