function TrainModel(pre, dim, nl, pretrainingFilename, testFilenames, splitFilenames, expName)

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
hyperParams.dim = dim;
hyperParams.numRelations = 7; 
hyperParams.topDepth = 1;
hyperParams.penultDim = 21;
hyperParams.lambda = 0.0001;
hyperParams.relations = relations;
if pre
    hyperParams.noPretraining = false;
else
    hyperParams.noPretraining = true;
end
hyperParams.minFunc = false;
hyperParams.showExamples = false;
hyperParams.showConfusions = false;
hyperParams.norm = 2;

% Nonlinearities
hyperParams.compNL = @Sigmoid;
hyperParams.compNLDeriv = @SigmoidDeriv; 
if strcmp(nl, 'S')
    hyperParams.classNL = @Sigmoid;
    hyperParams.classNLDeriv = @SigmoidDeriv;
else
    hyperParams.classNL = @LReLU;
    hyperParams.classNLDeriv = @LReLUDeriv;
end

disp(hyperParams)

[ theta, thetaDecoder ] = InitializeModel(size(wordMap, 1), hyperParams);

% minfunc options (not tuned)
global options
options.Method = 'lbfgs';
options.MaxFunEvals = 1000;
options.DerivativeCheck = 'off';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
options.OutputFcn = @Display;

% adaGradSGD options (partially tuned)
options.numPasses = 1000;
options.miniBatchSize = 32; % tuned-ish
options.lr = 0.05;

% adaGradSGD display options
options.testFreq = 4;
options.confusionFreq = 32; % should be a multiple of testfreq
options.examplesFreq = 32; % should be a multiple of testfreq
options.checkpointFreq = 8;
options.name = expName;
options.runName = 'pre';

% Add relation vector, so it can be ref'd in error reporting.
disp(options)
if nargin > 3 && ~isempty(pretrainingFilename)
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
        % Forget and repeat?
    else
        theta = adaGradSGD(theta, options, thetaDecoder, worddata, hyperParams);
    end
end
    
hyperParams.quiet = true;
  
% Evalaute on word pair data
[~, ~, preAcc, preConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, worddata, hyperParams);


disp('Word pair confusion, PER: ')
disp('tr:  #     =     >     <     |     ^     v')
disp(preConfusion)
disp(preAcc)

% Reset composition function for training
theta = ReinitializeCompositionLayer (theta, thetaDecoder, hyperParams);


% Load training data

listing = dir('data-2/*.tsv');
splitFilenames = {listing.name};

% allConstitFilenames = [allConstitFilenames, beyondQ];

% trainFilenames = allConstitFilenames;
% testInd = ismember(allConstitFilenames, testFilenames);
% trainFilenames(testInd) = [];   
[trainDataset, testDatasets] = ...
    LoadConstitDatasets({}, splitFilenames, {}, wordMap, relationMap);

% Train
disp('Training')
options.MaxFunEvals = 1000;
options.DerivativeCheck = 'off';
options.runName = 'tr';

trainDataset = Symmetrize(trainDataset);

if hyperParams.minFunc
    theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, trainDataset, hyperParams, testDatasets);
    % Forget and repeat?
else
    theta = adaGradSGD(theta, options, thetaDecoder, trainDataset, hyperParams, testDatasets);
end

% Evaluate on training data
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
