% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainJoinModel(expName, mbs, dim)
% The main training and testing script. The first arguments to the function
% have been tweaked quite a few times depending on what is being tuned.

addpath('..')
    
if nargin > 4
    mkdir(expName); 
else
    expName = '.';
end

[wordMap, relationMap, relations] = ...
    LoadTrainingData('data/medium_train.tsv');

% disp('Uninformativizing:');
% worddata = Uninformativize(worddata);

% Set up hyperparameters:

% The dimensionality of the word/phrase vectors.
hyperParams.dim = dim;

% The number of relations.
hyperParams.numRelations = 7; 

% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = 1;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = 45;

% Regularization coefficient.
hyperParams.lambda = 0.002;

% A vector of text relation labels.
hyperParams.relations = relations;

% Turn off to pretrain on a word pair dataset.
hyperParams.noPretraining = true;

% Use minFunc instead of SGD. Must be separately downloaded.
hyperParams.minFunc = false;

% Ignore. Modified every few iters.
hyperParams.showExamples = false; 
hyperParams.showConfusions = false;

% L1 v. L2 regularization
hyperParams.norm = 2;

% Use untied composition layer params.
hyperParams.untied = false; 

% Remove some portion of the training datasets
hyperParams.datasetsPortion = 1;
hyperParams.dataPortion = 1;

hyperParams.useThirdOrder = true; % For composition
hyperParams.useThirdOrderComparison = 1; % For comparison


% Nonlinearities.
hyperParams.compNL = @Sigmoid;
hyperParams.compNLDeriv = @SigmoidDeriv; 
nl = 'M';
if strcmp(nl, 'S')
    hyperParams.classNL = @Sigmoid;
    hyperParams.classNLDeriv = @SigmoidDeriv;
elseif strcmp(nl, 'M')
    hyperParams.classNL = @LReLU;
    hyperParams.classNLDeriv = @LReLUDeriv;
end

disp(hyperParams)

% Randomly initialize.
[ theta, thetaDecoder ] = InitializeModel(size(wordMap, 1), hyperParams);

% minfunc options
global options
options.Method = 'lbfgs';
options.MaxFunEvals = 25000;
options.DerivativeCheck = 'off';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
options.OutputFcn = @Display;

% AdaGradSGD learning options

% Rarely does anything interesting happen past 
% ~iteration ~200.
options.numPasses = 10000;
options.miniBatchSize = mbs;

% LR
options.lr = 0.05;

% AdaGradSGD display options

% How often (in full iterations) to run on test data.
options.testFreq = 1;

% How often to report confusion matrices. 
% Should be a multiple of testFreq.
options.confusionFreq = 8;

% How often to display which items are misclassified.
% Should be a multiple of testFreq.
options.examplesFreq = 32; 

% How often to save parameters to disk.
options.checkpointFreq = 8; 

% The name assigned to the current full run. Used in checkpoint naming.
options.name = expName; 

% The name assigned to the current call to AdaGradSGD. Used to contrast ...
% pretraining and training in checkpoint naming.
options.runName = 'pre';

% Reset the sum of squared gradients after this many iterations.
options.resetSumSqFreq = 10000; % Don't bother.

disp(options)

if nargin > 4 && ~isempty(pretrainingFilename)
    % Initialize parameters from disk
    clear 'theta'
    clear 'thetaDecoder'
    v = load(pretrainingFilename);
    theta = v.theta;
    thetaDecoder = v.thetaDecoder;
elseif ~hyperParams.noPretraining
    % Pretrain words
    disp('Pretraining')
    if hyperParams.minFunc
        theta = minFunc(@ComputeFullCostAndGrad, theta, options, ...
            thetaDecoder, worddata, hyperParams);
        % TODO: Forget and repeat?
    else
        theta = AdaGradSGD(theta, options, thetaDecoder, worddata, hyperParams);
    end
end

if ~hyperParams.noPretraining
    % Evalaute on word pair data
    [~, ~, preAcc, preConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, worddata, hyperParams);

    disp('Word pair confusion, PER: ')
    disp('tr:  #     =     >     <     |     ^     v')
    disp(preConfusion)
    disp(preAcc)
end

% Reset composition function for training
theta = ReinitializeCompositionLayer (theta, thetaDecoder, hyperParams);

% Choose which files to load in each category.
% listing = dir('data-4/*.tsv');
splitFilenames = {};
trainFilenames = {'./data/medium_train.tsv'};
testFilenames = {'./data/medium_test.tsv', ...
                 './data/medium_test_underivable.tsv'};

% splitFilenames = setdiff(splitFilenames, testFilenames);
hyperParams.firstSplit = size(testFilenames, 2) + 1;

if hyperParams.datasetsPortion < 1
    disp(length(splitFilenames))
    p = randperm(length(splitFilenames));
    splitFilenames = splitFilenames(p(1:round(hyperParams.datasetsPortion * length(splitFilenames))));
    disp(length(splitFilenames))
end
    
% Load training/test data
[trainDataset, testDatasets] = ...
    LoadConstitDatasets(trainFilenames, splitFilenames, ...
    testFilenames, wordMap, relationMap);
% trainDataset = Symmetrize(trainDataset);

if hyperParams.dataPortion < 1
    disp(length(trainDataset))
    p = randperm(length(trainDataset));
    trainDataset = trainDataset(p(1:round(hyperParams.dataPortion * length(trainDataset))));
    disp(length(trainDataset))
end

% Train
disp('Training')
options.MaxFunEvals = 10000;
options.DerivativeCheck = 'off';
options.runName = 'tr';

if hyperParams.minFunc
    % Set up minfunc
    addpath('../minFunc/minFunc/')
    addpath('../minFunc/minFunc/compiled/')
    addpath('../minFunc/minFunc/mex/')
    addpath('../minFunc/autoDif/')

    theta = minFunc(@ComputeFullCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, hyperParams, testDatasets);
    % TODO: Forget metadata and repeat?
else
    theta = AdaGradSGD(@ComputeFullCostAndGrad, theta, options, thetaDecoder, trainDataset, ...
        hyperParams, testDatasets);
end

end
