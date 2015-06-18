function modelState = AdaGradUpdate(modelState, options, grad, embGrad)

if modelState.step == 0
    modelState.sumSqGrad = zeros(size(modelState.theta));
    if length(embGrad) > 0
        % Set up a separate SubSqGrad tracker for the embeddings.
        modelState.sumSqEmbGrad = zeros(size(modelState.separateWordFeatures));
    end
end

% Do an AdaGrad-scaled parameter update
modelState.sumSqGrad = modelState.sumSqGrad + grad.^2;
modelState.theta = modelState.theta - modelState.lr * (grad ./ (sqrt(modelState.sumSqGrad) + options.adaEps));

assert(sum(isnan(modelState.theta)) == 0, 'NaNs in theta.');
assert(sum(isinf(modelState.theta)) == 0, 'Infs in theta.');

% Do an AdaGrad-scaled parameter update to the separate word features
if length(embGrad) > 0
    modelState.sumSqEmbGrad = modelState.sumSqEmbGrad + embGrad.^2;
    modelState.separateWordFeatures = modelState.separateWordFeatures - modelState.lr * (embGrad ./ (sqrt(modelState.sumSqEmbGrad) + options.adaEps));
end

end