function TrainModel(td, dataflag, pretrainingFilename, testFilenames, trainFilenames, expName)
% The main training+testing script. The first arguments to the function
% have been tweaked quite a few times depending on what is being tuned.
% Typically called from the command line as here:
%   echo "cd /user/sbowman/quant/; \
%   TrainModel(1, 'M', [], {}, {}, 'depth1-mixedNL')" | \
%   /afs/cs/software/bin/matlab_r2012b | tee depth1-mixedNL.txt


if nargin > 6
    mkdir(expName); 
else
    expName = '.';
end

[worddata, wordMap, relationMap, relations] = ...
    LoadTrainingData('wordpairs-v2.tsv');

% disp('Uninformativizing:');
% worddata = Uninformativize(worddata);

% Set up minfunc
addpath('minFunc/minFunc/')
addpath('minFunc/minFunc/compiled/')
addpath('minFunc/minFunc/mex/')
addpath('minFunc/autoDif/')

% Set up hyperparameters:
hyperParams.dim = 16;
hyperParams.numRelations = 7; 
hyperParams.topDepth = td;
hyperParams.penultDim = 45;
hyperParams.lambda = 0.0001;
hyperParams.relations = relations;
hyperParams.noPretraining = true;
hyperParams.minFunc = false;
hyperParams.showExamples = false;
hyperParams.showConfusions = false;
hyperParams.norm = 2;

% Nonlinearities
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
options.MaxFunEvals = 1000;
options.DerivativeCheck = 'off';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
options.OutputFcn = @Display;

% adaGradSGD options
options.numPasses = 1000; % Rarely does anything interesting happen past 
                          % ~iteration ~200.
options.miniBatchSize = 32;
options.lr = 0.01;

% adaGradSGD display options
options.testFreq = 4; % How often (in full iterations) to run on test data.
options.confusionFreq = 32; % How often to report confusion matrices. 
                            % Should be a multiple of testFreq.
options.examplesFreq = 32; % How often to display which items are 
                           % misclassified. Should be a multiple of
                           % testFreq.
options.checkpointFreq = 8; % How often to save parameters to disk
options.name = expName; % The name assigned to the current full run. 
                        % Used in checkpoint naming.
options.runName = 'pre'; % The name assigned to the current call to 
                         % adaGradSGD. Used to contrast pretraining and 
                         % training in checkpoint naming.
disp(options)

if nargin > 3 && ~isempty(pretrainingFilename)
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
        theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, worddata, hyperParams);
        % TODO: Forget and repeat?
    else
        theta = adaGradSGD(theta, options, thetaDecoder, worddata, hyperParams);
    end
end
    
hyperParams.quiet = true;
  
% Evalaute on word pair data
% [~, ~, preAcc, preConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, worddata, hyperParams);
% 
% disp('Word pair confusion, PER: ')
% disp('tr:  #     =     >     <     |     ^     v')
% disp(preConfusion)
% disp(preAcc)

% Reset composition function for training
theta = ReinitializeCompositionLayer (theta, thetaDecoder, hyperParams);


% Choose which files to load in each category.
listing = dir('data-4/*.tsv');
splitFilenames = {listing.name};
trainFilenames = {};
if strcmp(dataflag, 'one')
    testFilenames = {'MQ-most-no-bark.tsv'};
elseif strcmp(dataflag, 'sub') 
    testFilenames = {'MQ-most-no-bark.tsv', 'MQ-most-no-European.tsv', 'MQ-most-no-mobile.tsv'};
elseif strcmp(dataflag, 'class')
    listing = [dir('data-4/*no-most*'); dir('data-4/*most-no*')];
    testFilenames = {listing.name};
elseif strcmp(dataflag, 'splitall')
    splitFilenames = {listing.name};
elseif strcmp(dataflag, 'testall')
    trainFilenames = splitFilenames;
    splitFilenames = {};
end
splitFilenames = setdiff(splitFilenames, testFilenames);

% Load
[trainDataset, testDatasets] = ...
    LoadConstitDatasets(trainFilenames, splitFilenames, testFilenames, wordMap, relationMap);

% Train
disp('Training')
options.MaxFunEvals = 1000;
options.DerivativeCheck = 'off';
options.runName = 'tr';

trainDataset = Symmetrize(trainDataset);

if hyperParams.minFunc
    theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, trainDataset, hyperParams, testDatasets);
    % TODO: Forget and repeat?
else
    theta = adaGradSGD(theta, options, thetaDecoder, trainDataset, hyperParams, testDatasets);
end

% Done. Evaluate final model on training data.
% (Mid-run results are usually better.)
[~, ~, trAcc, trConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, trainDataset, hyperParams);

disp('Training confusion, PER: ')
disp('tr:  #     =     >     <     |     ^     v')
disp(trConfusion)
disp(trAcc)

[teAcc, teConfusion] = TestModel(theta, thetaDecoder, testDatasets, hyperParams);

% Print results for all three full datasets
disp('Word pair confusion, PER: ')
disp('tr:  #     =     >     <     |     ^     v')
disp(preConfusion)
disp(preAcc)

disp('Training confusion, PER: ')
disp('tr:  #     =     >     <     |     ^     v')
disp(trConfusion)
disp(trAcc)

disp('Test confusion, PER: ')
disp('tr:  #     =     >     <     |     ^     v')
disp(teConfusion)
disp(teAcc)

end
