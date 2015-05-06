function modelState = AdaGradUpdate(modelState, options, hyperParams, grad, embGrad)

if modelState.step == 0
    modelState.sumSqGrad = fZeros(size(modelState.theta), hyperParams.gpu);
    if length(embGrad) > 0
        % Set up a separate SumSqGrad tracker for the embeddings.
        modelState.sumSqEmbGrad = fZeros(size(modelState.separateWordFeatures), hyperParams.gpu && ~hyperParams.largeVocabMode);
    end
end

% Do an AdaGrad-scaled parameter update
modelState.sumSqGrad = modelState.sumSqGrad + grad.^2;
modelState.theta = modelState.theta - modelState.lr * (grad ./ (sqrt(modelState.sumSqGrad) + options.adaEps));

% Do an AdaGrad-scaled parameter update to the separate word features
if length(embGrad) > 0
    modelState.sumSqEmbGrad = modelState.sumSqEmbGrad + embGrad.^2;
    modelState.separateWordFeatures = modelState.separateWordFeatures - modelState.lr * (embGrad ./ (sqrt(modelState.sumSqEmbGrad) + options.adaEps));
end

end
