function TrainModel(filenamePrefix, pretrainEvals, trainEvals)

[worddata, wordMap, relationMap] = LoadTrainingData('wordpairs-v2.tsv');

% Set up minfunc
addpath('minFunc/minFunc/')
addpath('minFunc/minFunc/compiled/')
addpath('minFunc/minFunc/mex/')
addpath('minFunc/autoDif/')

% Set up hyperparameters:
hyperParams.dim = 2;
hyperParams.numRelations = 7;
hyperParams.penultDim = 4;
hyperParams.lambda = 0.00001;
hyperParams

% Load short-name variables.
DIM = hyperParams.dim;
PENULT_DIM = hyperParams.penultDim;

[ theta, thetaDecoder ] = InitializeModel(size(wordMap, 1), hyperParams);
worddata = InitializeWordFeatures(worddata, theta, thetaDecoder);

% minfunc options (not tuned)
options.Method = 'lbfgs';
options.MaxFunEvals = pretrainEvals;
options.DerivativeCheck = 'off';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
options

% Pretrain words
disp('Pretraining')
theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, worddata, hyperParams);
theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, worddata, hyperParams);
[cost, grad, preAcc, preConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, worddata, hyperParams);

disp('Word pair confusion, accuracy: ')
preConfusion
preAcc

% Reset composition function for training
theta = ReinitializeCompositionLayer (theta, thetaDecoder, hyperParams);

% Load training data
traindata = LoadConstitData(strcat(filenamePrefix, '-train.tsv'), wordMap, relationMap);
traindata = InitializeWordFeatures(traindata, theta, thetaDecoder);

disp('Training')
options.MaxFunEvals = trainEvals;

theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, worddata, hyperParams);
theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, worddata, hyperParams);
[cost, grad, trAcc, trConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, traindata, hyperParams);

% Evaluate on the test data
testdata = LoadConstitData(strcat(filenamePrefix,'-test.tsv'), wordMap, relationMap);
testdata = InitializeWordFeatures(testdata, theta, thetaDecoder);

[cost, grad, acc, confusion] = ComputeFullCostAndGrad(theta, thetaDecoder, testdata, hyperParams);

% Print results for all three cases
disp('Word pair confusion, accuracy: ')
preConfusion
preAcc

disp('Training confusion, accuracy: ')
trConfusion
trAcc

disp('Test confusion, accuracy: ')
confusion
acc



end
