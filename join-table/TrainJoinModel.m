% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainJoinModel(expName, mbs, dim, tot, lambda, penult, sig)
% The main training and testing script for the model that learns the 
% join table (Experiment 1). 

% Make the rest of the package within scope
addpath('..')

if nargin > 5
    mkdir(expName); 
else
    expName = '.';
end

% Open the log files and add them to the config object
hyperParams.statlog = fopen([expName '/stat_log'], 'a');
hyperParams.examplelog = fopen([expName '/example_log'], 'a');

[wordMap, relationMap, relations] = ...
    LoadTrainingData('./data/6x80_train.tsv')

% disp('Uninformativizing:');
% worddata = Uninformativize(worddata);

% Set up hyperparameters:

% The dimensionality of the word/phrase vectors.
hyperParams.dim = dim;

% If true, don't load the entire training dataset into memory at once. Not useful in the join model.
hyperParams.fragmentData = false;

% The name of the vocab used in naming stored preprocessed data files. Not useful in the join model.
hyperParams.vocabName = 'join';

% If true, initialize the vocabulary from disk. Not useful in the join model.
hyperParams.loadWords = false;

% The number of relations.
hyperParams.numRelations = [7]; 

% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = 1;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; %0.002;

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

% Remove some portion of the training datasets for data volume experiments.
hyperParams.datasetsPortion = 1;
hyperParams.dataPortion = 1;


hyperParams.embeddingTransformDepth = 0;
hyperParams.trainWords = 1;

% Whether to use a plain RNN or an RNTN
hyperParams.useThirdOrder = tot;
hyperParams.useThirdOrderComparison = tot;

% Nonlinearities.
hyperParams.compNL = @Sigmoid;
hyperParams.compNLDeriv = @SigmoidDeriv; 
if (sig)
    hyperParams.classNL = @Sigmoid;
    hyperParams.classNLDeriv = @SigmoidDeriv;
else
    hyperParams.classNL = @LReLU;
    hyperParams.classNLDeriv = @LReLUDeriv;
end

% Log the model configuration
Log(hyperParams.statlog, ['Model config: ' evalc('disp(hyperParams)')])

global options

if hyperParams.minFunc
    % minfunc options
    options.Method = 'lbfgs';
    options.MaxFunEvals = 25000;
    options.DerivativeCheck = 'off';
    options.Display = 'full';
    options.numDiff = 0;
    options.LS_init = '2'; % Attempt to minimize evaluations per step...
    options.PlotFcns = [];
    options.OutputFcn = @Display;
else
    % AdaGradSGD learning options
    options.numPasses = 10000;
    options.miniBatchSize = mbs;

% LR
options.lr = 0.2; % TODO...

% Display options

% How often (in steps) to report cost.
options.costFreq = 500;

% How often (in steps) to run on test data.
options.testFreq = 500;

% How often to report confusion matrices. 
% Should be a multiple of testFreq.
options.confusionFreq = 500;

% How often to display which items are misclassified.
% Should be a multiple of testFreq.
options.examplesFreq = 1000; 

% How often (in steps) to save parameters to disk.
options.checkpointFreq = 4000; 

% The name assigned to the current full run. Used in checkpoint naming.
options.name = expName; 

% The name assigned to the current call to AdaGradSGD. Used to distinguish
% pretraining and training in checkpoint naming.
options.runName = 'tr';

% Reset the sum of squared gradients after this many iterations.
% WARNING: The countdown to a reset will be restarted if the model dies
% and is reloaded from a checkpoint.
options.resetSumSqFreq = 100000; % Don't bother.

Log(hyperParams.statlog, ['Model training options: ' evalc('disp(options)')])

% Load saved parameters if available
savedParams = '';
if nargin > 7 && ~isempty(pretrainingFilename)
    savedParams = pretrainingFilename;
else
    listing = dir([options.name, '/', 'ckpt*']);
    if ~isempty(listing)
        savedParams = [options.name, '/', listing(end).name];
    end
end
if ~isempty(savedParams)
    Log(hyperParams.statlog, ['Loading parameters: ' savedParams]);
    a = load(savedParams);
    modelState = a.modelState;
else
    modelState.step = 0;
    Log(hyperParams.statlog, ['Randomly initializing.']);
    [ modelState.theta, modelState.thetaDecoder ] = ...
       InitializeModel(wordMap, hyperParams);
end

% Choose which files to load in each category.
splitFilenames = {};
trainFilenames = {'./data/6x80_train.tsv'};
testFilenames = {'./data/6x80_test.tsv', ...
                 './data/6x80_test_underivable.tsv'}; % TODO, check dir!

% splitFilenames = setdiff(splitFilenames, testFilenames);
hyperParams.firstSplit = size(testFilenames, 2) + 1;

if hyperParams.datasetsPortion < 1
    p = randperm(length(splitFilenames));
    splitFilenames = splitFilenames(p(1:round(hyperParams.datasetsPortion * length(splitFilenames))));
end
    
% Load training/test data
[trainDataset, testDatasets] = ...
    LoadConstitDatasets(trainFilenames, splitFilenames, ...
    testFilenames, wordMap, relationMap, hyperParams);
% trainDataset = Symmetrize(trainDataset);

if hyperParams.dataPortion < 1
    p = randperm(length(trainDataset));
    trainDataset = trainDataset(p(1:round(hyperParams.dataPortion * length(trainDataset))));
end

% Train
Log(hyperParams.statlog, 'Training.');

if hyperParams.minFunc
    % Set up minfunc
    addpath('../minFunc/minFunc/')
    addpath('../minFunc/minFunc/compiled/')
    addpath('../minFunc/minFunc/mex/')
    addpath('../minFunc/autoDif/')

    theta = minFunc(@ComputeFullCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, hyperParams, testDatasets);
else
    theta = AdaGradSGD(@ComputeFullCostAndGrad, modelState, options, trainDataset, ...
        hyperParams, testDatasets);
end

end
