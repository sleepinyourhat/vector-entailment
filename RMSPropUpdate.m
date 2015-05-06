% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function modelState = RMSPropUpdate(modelState, options, hyperParams, grad, embGrad)

% Based on the Lua implementation in Kai Sheng Tai's torch-ntm:
% https://github.com/kaishengtai/torch-ntm/blob/master/rmsprop.lua

if modelState.step == 0
    modelState.gradAccum = fZeros(size(modelState.theta), hyperParams.gpu);
    modelState.sqGradAccum = fZeros(size(modelState.theta), hyperParams.gpu);
    modelState.update = fZeros(size(modelState.theta), hyperParams.gpu);
    if length(embGrad) > 0
        % Set up a separate tracker for the embeddings.
        modelState.embGradAccum = fZeros(size(modelState.separateWordFeatures), hyperParams.gpu && ~hyperParams.largeVocabMode);
        modelState.embSqGradAccum = fZeros(size(modelState.separateWordFeatures), hyperParams.gpu && ~hyperParams.largeVocabMode);
        modelState.embUpdate = fZeros(size(modelState.separateWordFeatures), hyperParams.gpu && ~hyperParams.largeVocabMode);
    end
end

modelState.gradAccum = modelState.gradAccum .* options.RMSPropDecay + ...
                       grad .* (1 - options.RMSPropDecay);

modelState.sqGradAccum = modelState.sqGradAccum .* options.RMSPropDecay + ...
                         (grad .^ 2) .* (1 - options.RMSPropDecay);                 

modelState.update = modelState.update .* options.momentum - ...
                    options.lr .* (grad ./ ...
                    sqrt(modelState.sqGradAccum - (modelState.gradAccum .^ 2) + options.RMSPropEps)); 

modelState.theta = modelState.theta + modelState.update;

% Do a scaled parameter update to the separate word features.
if length(embGrad) > 0
    modelState.embGradAccum = modelState.embGradAccum .* options.RMSPropDecay + ...
                           embGrad .* (1 - options.RMSPropDecay);

    modelState.embSqGradAccum = modelState.embSqGradAccum .* options.RMSPropDecay + ...
                             (embGrad .^ 2) .* (1 - options.RMSPropDecay);                 

    modelState.embUpdate = modelState.embUpdate .* options.momentum - ...
                        options.lr .* (embGrad ./ ...
                        sqrt(modelState.embSqGradAccum - (modelState.embGradAccum .^ 2) + options.RMSPropEps)); 

    modelState.separateWordFeatures = modelState.separateWordFeatures + modelState.embUpdate;
end

end
