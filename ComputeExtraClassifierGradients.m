% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [matrixStackGradients, ...
          biasStackGradients, deltaDown] = ...
          ComputeExtraClassifierGradients(matrixStack, ...
              deltaDown, inputs, innerOutputs, classNLDeriv)
% Compute gradients for the middle NN layers of the classifier. This will 
% only do non-trivial work if STACKSIZE is greater than 0.

DIM = size(matrixStack, 1);
STACKSIZE = size(matrixStack, 3);

matrixStackGradients = zeros(DIM, DIM, STACKSIZE);
biasStackGradients = zeros(DIM, STACKSIZE);

for layer = (STACKSIZE):-1:1
    NLDeriv = classNLDeriv(innerOutputs(:, layer));

    % Calculate matrix gradients
    for i = 1:DIM
        matrixStackGradients(i, :, layer) = (NLDeriv(i) * deltaDown(i)) ...
            .* inputs(:, layer);
    end

    % Calculate bias gradients
    biasStackGradients(:, layer) = (NLDeriv .* deltaDown);

    % Calculate deltas to pass down
    thirdTerm = matrixStack(:, :, layer)';
    deltaDown = (thirdTerm * (biasStackGradients(:, layer) .* NLDeriv));
end
    
end