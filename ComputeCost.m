function [ cost, correct ] = ComputeCost( theta, decoder, dataPoint, hyperParams )
%function [ cost, grad ] = ComputeCostAndGrad( theta, decoder, dataPoint, hyperParams )
%   Detailed explanation goes here

% Unpack theta.
[classifierMatrices, classifierMatrix, classifierBias, ...
    classifierParameters, wordFeatures] = stack2param(theta, decoder);

% Unpack hyperparams.
NUM_RELATIONS = hyperParams.numRelations;
PENULT_DIM = hyperParams.penultDim;
DIM = hyperParams.dim;

leftTree = dataPoint.leftTree;
rightTree = dataPoint.rightTree;
trueRelation = dataPoint.relation;

% Make sure word features are current.
leftTree.updateFeatures(wordFeatures);
rightTree.updateFeatures(wordFeatures);

leftFeatures = leftTree.getFeatures();
rightFeatures = rightTree.getFeatures();

% Use the tensor layer to build classifier input:
tensorInnerOutput = ComputeInnerTensorLayer(leftFeatures, rightFeatures, classifierMatrices, classifierMatrix, classifierBias);
tensorOutput = Sigmoid(tensorInnerOutput);
tensorDeriv = SigmoidDeriv(tensorInnerOutput);

relationProbs = ComputeSoftmaxProbabilities(tensorOutput, classifierParameters);

% Increment local error
cost = Objective(trueRelation, relationProbs);

end

