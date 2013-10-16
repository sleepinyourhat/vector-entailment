function TrainModel(filenamePrefix, pretrainEvals, trainEvals)

[worddata, wordMap, relationMap] = LoadTrainingData('wordpairs-v2.tsv');

% Set up minfunc
addpath('minFunc/minFunc/')
addpath('minFunc/minFunc/compiled/')
addpath('minFunc/minFunc/mex/')
addpath('minFunc/autoDif/')

% Set up hyperparameters:
hyperParams.dim = 12;
hyperParams.numRelations = 7;
hyperParams.penultDim = 25;
hyperParams.lambda = 0.000001;
hyperParams

% Load short-name variables.
DIM = hyperParams.dim;
PENULT_DIM = hyperParams.penultDim;

[ theta, thetaDecoder ] = InitializeModel(size(wordMap, 1), hyperParams);

% minfunc options (not tuned)
options.Method = 'lbfgs';
options.MaxFunEvals = pretrainEvals;
options.DerivativeCheck = 'off';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
% Add relation vector, so it can be ref'd in error reporting.
disp(options)

% Pretrain words
disp('Pretraining')
theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, worddata, hyperParams);
% Forget and repeat?
[cost, grad, preAcc, preConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, worddata, hyperParams);

disp('Word pair confusion, accuracy: ')
disp(preConfusion)
disp(preAcc)

% Reset composition function for training
theta = ReinitializeCompositionLayer (theta, thetaDecoder, hyperParams);

% Load training data
traindata = LoadConstitData(strcat(filenamePrefix, '-train.tsv'), wordMap, relationMap);

disp('Training')
options.MaxFunEvals = trainEvals;
options.DerivativeCheck = 'off';


theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, traindata, hyperParams);
% Forget and repeat?
[cost, grad, trAcc, trConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, traindata, hyperParams);

% Evaluate on the test data
testdata = LoadConstitData(strcat(filenamePrefix,'-test.tsv'), wordMap, relationMap);

[cost, grad, acc, confusion] = ComputeFullCostAndGrad(theta, thetaDecoder, testdata, hyperParams);

% Print results for all three cases
disp('Word pair confusion, accuracy: ')
disp(preConfusion)
disp(preAcc)

disp('Training confusion, accuracy: ')
disp(trConfusion)
disp(trAcc)

disp('Test confusion, accuracy: ')
disp(confusion)
disp(acc)



end
