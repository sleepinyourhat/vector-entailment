% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainLatticeModel(expName, dim, mbs, tot)
% The main training and testing script. The first arguments to the function
% have been tweaked quite a few times depending on what is being tuned.

addpath('..')

if nargin > 4
    mkdir(expName); 
else
    expName = '.';
end

[wordMap] = ...
    LoadLatticeVocabulary('../join-algebra/powerset_2_meet_join_complete_test.txt');

% disp('Uninformativizing:');
% worddata = Uninformativize(worddata);

% Set up hyperparameters:

% Whether the trees are annotated with relation symbols
hyperParams.treeMode = 'meet-join';

% The dimensionality of the word/phrase vectors.
hyperParams.dim = dim;

% The number of classes.
hyperParams.numRelations = wordMap.size(1);

% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = 2;

% The dimensionality of the comparison layer(s).
% Must equal dim for now without a reduction layer.
hyperParams.penultDim = hyperParams.dim;

% Regularization coefficient.
hyperParams.lambda = 0.002;

% Use minFunc instead of SGD. Must be separately downloaded.
hyperParams.minFunc = false;

% Ignore. Modified every few iters.
hyperParams.showExamples = false; 
hyperParams.showConfusions = false;

% L1 v. L2 regularization
hyperParams.norm = 2;

% Remove some portion of the training datasets
hyperParams.datasetsPortion = 1;
hyperParams.dataPortion = 1;

hyperParams.useThirdOrder = tot; % For composition

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
[ theta, thetaDecoder ] = InitializeMeetJoinModel(hyperParams.numRelations, hyperParams);

% minfunc options
global options
options.Method = 'lbfgs';
options.MaxFunEvals = 25000;
options.DerivativeCheck = 'off';
options.Display = 'iter';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
% options.OutputFcn = @Display;

% AdaGradSGD learning options

% Rarely does anything interesting happen past 
% ~iteration ~200.
options.numPasses = 1000;
options.miniBatchSize = mbs;

% LR
options.lr = 0.1;

% AdaGradSGD display options

% How often (in full iterations) to run on test data.
options.testFreq = 1;

% How often to report confusion matrices. 
% Should be a multiple of testFreq.
options.confusionFreq = 32;

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
end

% Choose which files to load in each category.
% listing = dir('data-4/*.tsv');
% splitFilenames = {};
% trainFilenames = {'../join-algebra/powerset_2_meet_join_complete_train.txt'};
% testFilenames = {'../join-algebra/powerset_2_meet_join_complete_test.txt'};

% splitFilenames = setdiff(splitFilenames, testFilenames);
hyperParams.firstSplit = 2;

if hyperParams.datasetsPortion < 1
    disp(length(splitFilenames))
    p = randperm(length(splitFilenames));
    splitFilenames = splitFilenames(p(1:round(hyperParams.datasetsPortion * length(splitFilenames))));
    disp(length(splitFilenames))
end
    
% Load training/test data
% [trainDataset, testDatasets] = ...
%    LoadConstitDatasets(trainFilenames, splitFilenames, ...
%    testFilenames, wordMap, relationMap);
% trainDataset = Symmetrize(trainDataset);

trainDataset = LoadMeetJoinData('../join-algebra/2-lattice-d2-train.txt', wordMap);
testDatasets = {{'test'}, {LoadMeetJoinData('../join-algebra/2-lattice-d2-test.txt', wordMap)}};

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

    theta = minFunc(@ComputeFullLatticeCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, hyperParams, testDatasets);
    % TODO: Forget metadata and repeat?
else
    theta = AdaGradSGD(@ComputeFullLatticeCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, hyperParams, testDatasets);
end

% Done. Evaluate final model on training data.
% (Mid-run results are usually better.)
[~, ~, trAcc, trConfusion] = ComputeFullCostAndGrad(theta, ...
    thetaDecoder, trainDataset, hyperParams);

[teAcc, teConfusion] = TestModel(theta, thetaDecoder, testDatasets, ...
    hyperParams);

end
