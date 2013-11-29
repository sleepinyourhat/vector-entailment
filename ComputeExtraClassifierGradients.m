function [extraMatrixGradients, ...
          extraBiasGradients, deltaDown] = ...
          ComputeExtraClassifierGradients(hyperParams, ...
          classifierExtraMatrix, deltaDown, inputs, innerOutputs)
% Compute gradients for the non-bottom layers of the classifier.

outDim = size(classifierExtraMatrix);

extraMatrixGradients = zeros(hyperParams.penultDim, ...
                             hyperParams.penultDim, hyperParams.topDepth - 1);
extraBiasGradients = zeros(hyperParams.penultDim, hyperParams.topDepth - 1);

for layer = (hyperParams.topDepth - 1):-1:1
    NLDeriv = hyperParams.classNLDeriv(innerOutputs(:,layer));

    % Calculate matrix gradients
    for i = 1:outDim
        extraMatrixGradients(i,:,layer) = (NLDeriv(i) * deltaDown(i)) ...
            .* inputs(:,layer);
    end

    % Calculate bias gradients
    extraBiasGradients(:,layer) = (NLDeriv .* deltaDown);

    % Calculate deltas to pass down
    thirdTerm = classifierExtraMatrix(:, :, layer)';
    deltaDown = (thirdTerm * (extraBiasGradients(:,layer) .* NLDeriv));
end
    
end