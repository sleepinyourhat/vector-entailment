% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainModel(dataflag, pretrainingFilename, expName, mbs, dim, lr, lambda, tot)
% The main training and testing script. The first arguments to the function
% have been tweaked quite a few times depending on what is being tuned.

if nargin > 4
    mkdir(expName); 
else
    expName = '.';
end

if strcmp(dataflag, 'and-or') ||  strcmp(dataflag, 'and-or-deep') ||  strcmp(dataflag, 'and-or-deep-unlim')
    [wordMap, relationMap, relations] = ...
        LoadTrainingData('./RC/train1'); 
elseif findstr(dataflag, 'G-')
    [wordMap, relationMap, relations] = ...
        LoadTrainingData('./grammars/wordlist.tsv'); 
else
    [wordMap, relationMap, relations] = ...
        LoadTrainingData('./wordpairs-v2.tsv');
end

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
hyperParams.penultDim = 75;

% Regularization coefficient.
hyperParams.lambda = lambda %0.002;

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

hyperParams.useThirdOrder = tot;
hyperParams.useThirdOrderComparison = tot;

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

% minfunc options
global options
options.Method = 'lbfgs';
options.MaxFunEvals = 1000;
options.DerivativeCheck = 'on';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
options.OutputFcn = @Display;

% AdaGradSGD learning options

% Rarely does anything interesting happen past 
% ~iteration ~200.
options.numPasses = 1000;
options.miniBatchSize = mbs;

% LR
options.lr = lr;    % 0.2;

% AdaGradSGD display options

% How often (in full iterations) to run on test data.
options.testFreq = 1;

% How often to report confusion matrices. 
% Should be a multiple of testFreq.
options.confusionFreq = 1;

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

% Choose which files to load in each category.
listing = dir('data-4/*.tsv');
listing5 = dir('data-5/*.tsv');
listingG = dir('grammars/data/quant*');

splitFilenames = {listing.name};
trainFilenames = {};
testFilenames = {};

if strcmp(dataflag, 'sub-tn')
    testFilenames = {'MQ-three-no-bark.tsv', 'MQ-three-no-European.tsv', ...
        'MQ-three-no-mobile.tsv'};
    splitFilenames = {listing5.name};
elseif strcmp(dataflag, 'pair-tn')
    listing = [dir('data-5/*three-no*'); dir('data-5/*no-three*')];
    testFilenames = {'MQ-three-no-bark.tsv', listing.name};
    splitFilenames = {listing5.name};
elseif strcmp(dataflag, 'sub-nmn')
    testFilenames = {'NEG-MQ-R2-most-no-bark.tsv'};
    splitFilenames = {listing5.name};
elseif strcmp(dataflag, 'pair-nmn')
    listing = [dir('data-5/*most-no*'); dir('data-5/*no-most*')];
    testFilenames = {'NEG-MQ-R2-most-no-bark.tsv', listing.name};
    splitFilenames = {listing5.name};
elseif strcmp(dataflag, 'sub-ts')
    testFilenames = {'MT-MQ-two-some-2-French-Parisian-rev.tsv'};
    splitFilenames = {listing5.name};
elseif strcmp(dataflag, 'pair-ts')
    listing = [dir('data-5/*two-some*'); dir('data-5/*some-two*')];
    testFilenames = {'MT-MQ-two-some-2-French-Parisian-rev.tsv', listing.name};
    splitFilenames = {listing5.name};

elseif strcmp(dataflag, 'G-two_lt_two')
    testFilenames = {'grammars/data/quant_two_lt_two'};
    splitFilenames = {listingG.name};
elseif strcmp(dataflag, 'G-no_no')
    testFilenames = {'grammars/data/quant_no_no'};
    splitFilenames = {listingG.name};
elseif strcmp(dataflag, 'G-not_all_not_most')
    testFilenames = {'grammars/data/quant_not_all_not_most'};
    splitFilenames = {listingG.name};
elseif strcmp(dataflag, 'G-all_some')
    testFilenames = {'grammars/data/quant_all_some'};
    splitFilenames = {listingG.name};
elseif strcmp(dataflag, 'G-splitall')
    splitFilenames = {listingG.name};

elseif strcmp(dataflag, 'splitall-5')
    splitFilenames = {listing5.name};
elseif strcmp(dataflag, 'one-mn')
    testFilenames = {'MQ-most-no-bark.tsv'};
elseif strcmp(dataflag, 'sub-mn')
    testFilenames = {'MQ-most-no-bark.tsv', 'MQ-most-no-European.tsv', ...
        'MQ-most-no-mobile.tsv'};
elseif strcmp(dataflag, 'pair-mn')
    listing = [dir('data-4/*no-most*'); dir('data-4/*most-no*')];
    testFilenames = {listing.name};
elseif strcmp(dataflag, 'one-sn')
    testFilenames = {'BQ-some-no-bark.tsv'};
elseif strcmp(dataflag, 'sub-sn') 
    testFilenames = {'BQ-some-no-bark.tsv', 'BQ-some-no-European.tsv', ...
        'BQ-some-no-mobile.tsv'};
elseif strcmp(dataflag, 'pair-sn')
    listing = [dir('data-4/*some-no*'); dir('data-4/*no-some*')];
    testFilenames = {listing.name};
elseif strcmp(dataflag, 'one-2a')
    testFilenames = {'MQ-two-all-bark.tsv'};
elseif strcmp(dataflag, 'sub-2a') 
    testFilenames = {'MQ-two-all-bark.tsv', 'MQ-two-all-European.tsv', ...
        'MQ-two-all-mobile.tsv'};
elseif strcmp(dataflag, 'pair-2a')
    listing = [dir('data-4/*two-all*'); dir('data-4/*all-two*')];
    testFilenames = {listing.name};
elseif strcmp(dataflag, 'splitall')
    splitFilenames = {listing.name};
elseif strcmp(dataflag, 'testall')
    trainFilenames = splitFilenames;
    splitFilenames = {};
elseif strcmp(dataflag, 'gradcheck')
    splitFilenames = {'MQ-two-all-bark.tsv'};
    hyperParams.dim = 2;
    hyperParams.penultDim = 2;
    hyperParams.minFunc = 1;
elseif strcmp(dataflag, 'test')
    splitFilenames = {'MQ-two-all-bark.tsv'};
elseif strcmp(dataflag, 'and-or') 
    testFilenames = {'./RC/test0', './RC/test1', './RC/test2', './RC/test3', './RC/test4', './RC/test5', './RC/test6'};
    trainFilenames = {'./RC/train0', './RC/train1', './RC/train2', './RC/train3', './RC/train4'};
    splitFilenames = {};
    options.numPasses = 15000;
    if ~isempty(pretrainingFilename)
        hyperParams.penultDim = 45;
    end
elseif strcmp(dataflag, 'and-or-deep') 
    testFilenames = {'./RC/longer2/test0', './RC/longer2/test1', './RC/longer2/test2', './RC/longer2/test3', './RC/longer2/test4', './RC/longer2/test5', './RC/longer2/test6', './RC/longer2/test7', './RC/longer2/test8', './RC/longer2/test9', './RC/longer2/test10', './RC/longer2/test11', './RC/longer2/test12'};
    trainFilenames = {'./RC/longer2/train0', './RC/longer2/train1', './RC/longer2/train2', './RC/longer2/train3', './RC/longer2/train4'};
    splitFilenames = {};
    options.numPasses = 15000;
    if ~isempty(pretrainingFilename)
        hyperParams.penultDim = 45;
    end
elseif strcmp(dataflag, 'and-or-deep-unlim') 
    testFilenames = {'./RC/longer2/test0', './RC/longer2/test1', './RC/longer2/test2', './RC/longer2/test3', './RC/longer2/test4', './RC/longer2/test5', './RC/longer2/test6', './RC/longer2/test7', './RC/longer2/test8', './RC/longer2/test9', './RC/longer2/test10', './RC/longer2/test11', './RC/longer2/test12'};
    trainFilenames = {'./RC/longer2/train0', './RC/longer2/train1', './RC/longer2/train2', './RC/longer2/train3', './RC/longer2/train4', './RC/longer2/train5', './RC/longer2/train6', './RC/longer2/train7', './RC/longer2/train8', './RC/longer2/train9', './RC/longer2/train10', './RC/longer2/train11', './RC/longer2/train12'};
    splitFilenames = {};
    options.numPasses = 15000;
    if ~isempty(pretrainingFilename)
        hyperParams.penultDim = 45;
    end
end
splitFilenames = setdiff(splitFilenames, testFilenames);
hyperParams.firstSplit = 3;

if hyperParams.datasetsPortion < 1
    disp(length(splitFilenames))
    p = randperm(length(splitFilenames));
    splitFilenames = splitFilenames(p(1:round(hyperParams.datasetsPortion * length(splitFilenames))));
    disp(length(splitFilenames))
end

    
if ~isempty(pretrainingFilename)
    a = load(pretrainingFilename);
    theta = a.theta;
    thetaDecoder = a.thetaDecoder;
else 
    % Randomly initialize.
    [ theta, thetaDecoder ] = InitializeModel(size(wordMap, 1), hyperParams);
end

% Load training/test data
[trainDataset, testDatasets] = ...
    LoadConstitDatasets(trainFilenames, splitFilenames, ...
    testFilenames, wordMap, relationMap);
trainDataset = Symmetrize(trainDataset);



if hyperParams.dataPortion < 1
    disp(length(trainDataset))
    p = randperm(length(trainDataset));
    trainDataset = trainDataset(p(1:round(hyperParams.dataPortion * length(trainDataset))));
    disp(length(trainDataset))
end

% Train
disp('Training')
options.MaxFunEvals = 1000;
options.DerivativeCheck = 'on';
options.runName = 'tr';

if hyperParams.minFunc
    % Set up minfunc
    addpath('minFunc/minFunc/')
    addpath('minFunc/minFunc/compiled/')
    addpath('minFunc/minFunc/mex/')
    addpath('minFunc/autoDif/')

    theta = minFunc(@ComputeFullCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, hyperParams, testDatasets);
    % TODO: Forget metadata and repeat?
else
    theta = AdaGradSGD(@ComputeFullCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, ...
        hyperParams, testDatasets);
end

if strcmp(dataflag, 'and-or') 
    testFilenames = {};
    trainFilenames = {'./RC/extra-long-med.tsv'};
    splitFilenames = {};

    % Load training/test data
    [trainDataset, ~] = ...
        LoadConstitDatasets(trainFilenames, splitFilenames, ...
        testFilenames, wordMap, relationMap);
    trainDataset = Symmetrize(trainDataset);

    options.numPasses = 250;
    
    theta = AdaGradSGD(@ComputeFullCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, ...
        hyperParams, testDatasets);
        
    options.numPasses = 1000;
    
    testFilenames = {};
    trainFilenames = {'./RC/extra-long-train.tsv'};
    splitFilenames = {};

    [trainDataset, ~] = ...
        LoadConstitDatasets(trainFilenames, splitFilenames, ...
        testFilenames, wordMap, relationMap);
    trainDataset = Symmetrize(trainDataset);

    theta = AdaGradSGD(@ComputeFullCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, ...
        hyperParams, testDatasets);
end
    
end
