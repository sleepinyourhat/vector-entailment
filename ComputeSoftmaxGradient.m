function [softmaxGradient, softmaxDelta] = ...
    ComputeSoftmaxGradient (hyperParams, classifierParameters, ...
                            relationProbs, trueRelation, tensorOutput)
% Compute the gradient for the softmax layer parameters, and the deltas to
% pass down.
                        
softmaxGradient = zeros(hyperParams.numRelations, ...
    hyperParams.penultDim + 1);

% Compute node softmax error, mid-left of p6 of Socher tensor paper
targetRelationProbs = zeros(length(relationProbs), 1);
targetRelationProbs(trueRelation) = 1;
softmaxDeltaFirstHalf = classifierParameters' * ...
                        (relationProbs - targetRelationProbs);
softmaxDeltaSecondHalf = hyperParams.classNLDeriv([1; tensorOutput]); % Intercept
softmaxDelta = (softmaxDeltaFirstHalf .* softmaxDeltaSecondHalf);

for relEval = 1:hyperParams.numRelations
    % Del from ufldl wiki on softmax
    softmaxGradient(relEval, :) = -([1; tensorOutput] .* ...
        ((trueRelation == relEval) - relationProbs(relEval)))';
end

softmaxDelta = softmaxDelta(2:hyperParams.penultDim+1);

end