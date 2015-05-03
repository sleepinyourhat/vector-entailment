'DEPRECATED'

DIM = 2;
PENULT = 2;
SCALE = 0.05;
WLSTM = rand(DIM * 4, DIM * 2 + 1, 1) .* (2 * SCALE) - SCALE;

  % compositionMatrix(2 * DIM + 1:3 * DIM, 1) = 1;
  % compositionMatrix(DIM + 1:2 * DIM, 1) = 1;
  % compositionMatrix(3 * DIM + 1:4 * DIM, 1) = 1;

addpath('config/')
addpath('layer-fns/')

map = containers.Map({'a', 'b'}, {1, 2});
sl = Sequence.makeSequence('b a a a a a', map, 1);
sr = Sequence.makeSequence('a a a a a a', map, 1);

dataPoint(1).left = sl;
dataPoint(1).right = sr;
dataPoint(1).label = 1;

dataPoint(2).left = sl;
dataPoint(2).right = sr;
dataPoint(2).label = 1;

hyperParams = GradCheck(0, 1, 0, 0, 1, 0, 0, 1, 0);

[ theta, dec, separateWordFeatures ] = InitializeModel(map, hyperParams);

[ ~, grad ] = ComputeFullCostAndGrad(theta, dec, dataPoint, separateWordFeatures, hyperParams, 1);

numGrad = 0 .* theta;

epsi = 1e-8;
for i = 1:length(theta)
	tempTheta = theta;
	tempTheta(i) = theta(i) + epsi;
	[ p ] = ComputeFullCostAndGrad(tempTheta, dec, dataPoint, separateWordFeatures, hyperParams, 0);

	tempTheta(i) = theta(i) - epsi;
	[ m ] = ComputeFullCostAndGrad(tempTheta, dec, dataPoint, separateWordFeatures, hyperParams, 0);

	numGrad(i) = ((p(1) - m(1)) ./ (2 * epsi));
end


[mergeMatrices, mergeMatrix, classifierBias, ...
    softmaxMatrix, trainedWordFeatures, compositionMatrices,...
    compositionMatrix, compositionBias, classifierExtraMatrix, ...
    classifierExtraBias, embeddingTransformMatrix, embeddingTransformBias] ...
    = stack2param(grad, dec);

[mergeMatrices, NmergeMatrix, classifierBias, ...
    softmaxMatrix, NtrainedWordFeatures, compositionMatrices,...
    NcompositionMatrix, compositionBias, classifierExtraMatrix, ...
    classifierExtraBias, embeddingTransformMatrix, embeddingTransformBias] ...
    = stack2param(numGrad, dec);

compositionMatrix

NcompositionMatrix

scaledWord = abs(compositionMatrix - NcompositionMatrix) ./ (abs(NcompositionMatrix) + abs(compositionMatrix) + eps) .* (abs(NcompositionMatrix) + abs(compositionMatrix) > 1e-7)

trainedWordFeatures
NtrainedWordFeatures

scaledWord = abs(trainedWordFeatures - NtrainedWordFeatures) ./ (abs(NtrainedWordFeatures) + abs(trainedWordFeatures) + eps) .* (abs(NtrainedWordFeatures) + abs(trainedWordFeatures) > 1e-7)
