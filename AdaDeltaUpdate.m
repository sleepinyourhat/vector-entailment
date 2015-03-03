function modelState = AdaDeltaUpdate(modelState, options, grad, embGrad)

COUNTING_SPAN = 1000;

if modelState.step == 0
    modelState.sumSqGrad = zeros(size(modelState.theta));
    modelState.sumSqDelta = zeros(size(modelState.theta));
    modelState.sumGradNorm = 0;
    modelState.meanGradNorm = -1;  % Will be filled in during run.
    if length(embGrad) > 0
        % Set up a separate SubSqGrad tracker for the embeddings.
        modelState.sumSqEmbGrad = zeros(size(modelState.separateWordFeatures));
        modelState.sumSqEmbDelta = zeros(size(modelState.separateWordFeatures));
    end
end

% Clip the gradient.
if options.clipGradients
    gradNorm = norm(grad);
    if modelState.meanGradNorm > 0 && gradNorm > modelState.meanGradNorm
        grad = grad ./ gradNorm;
    else
        modelState.sumGradNorm = modelState.sumGradNorm + gradNorm; %% TEMP:
        if modelState.step == COUNTING_SPAN - 1;
            modelState.meanGradNorm = modelState.sumGradNorm / COUNTING_SPAN;
                modelState
        end
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
    % Clip the embedding gradient, thresholding with the normal gradient norm.
    if options.clipGradients && modelState.meanGradNorm > 0 && gradNorm > modelState.meanGradNorm
        embGrad = embGrad ./ norm(embGrad);
    end

    modelState.sumSqEmbGrad = modelState.sumSqEmbGrad * options.adaDeltaRho + (embGrad.^2) * (1 - options.adaDeltaRho);

    rmsLastEmbDelta = sqrt(modelState.sumSqEmbDelta + options.adaDeltaEps);
    rmsEmbGrad = sqrt(modelState.sumSqEmbGrad + options.adaDeltaEps);

    embDelta = -1 * (rmsLastEmbDelta ./ rmsEmbGrad) .* embGrad;
    modelState.sumSqEmbDelta = modelState.sumSqEmbDelta * options.adaDeltaRho + (embDelta.^2) * (1 - options.adaDeltaRho);

    modelState.separateWordFeatures = modelState.separateWordFeatures + embDelta;
end

end