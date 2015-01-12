% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainModel(pretrainingFilename, fold, ConfigFn, varargin)
% The main training and testing script.
% Arguments:
%% pretrainingFilename: Pass in the filename to a checkpoint file to start training
%%%% from that checkpoint. If this argument is blank but a checkpoint with the exact same
%%%% experiment name is found, that checkpoint will be loaded. To start a fresh experiment,
%%%% use a fresh experiment name (set in the config file).
%% fold: Used for five fold cross-validation. Settings other than 1 will cause f# to be
%%%% appended to the experiment name.
%% ConfigFn: A function handle to an experiment configuration function that sets up 
%%%% hyperParams, options, wordMap, and relationMap. See examples in the config/ directory.
%% varargin: All remaining arguments will be passed through to the config function.


% Look for experiment configuration scripts in the config/ directory.
addpath('config/')

% Set up paralellization
c = parcluster();
t = tempname();
mkdir(t);
c.JobStorageLocation = t;
if exist('parpool')
  % >= 2013b
  parpool(c);
else
  % < 2013b
  matlabpool(c, c.NumWorkers);
end

[ hyperParams, options, wordMap, relationMap ] = ConfigFn(varargin{:});

% If the fold number is grater than one, the train/test split on split data will 
% be offset accordingly.
hyperParams.foldNumber = fold;

% The name assigned to the current full run. Used in checkpoint naming, and must
% match the directory created above.
if fold > 1
    hyperParams.name = [hyperParams.name, 'f', num2str(fold)];
end
options.name = hyperParams.name;

% Set up an experimental directory.
mkdir(hyperParams.name); 

hyperParams.statlog = fopen([hyperParams.name '/stat_log'], 'a');
hyperParams.examplelog = fopen([hyperParams.name '/example_log'], 'a');

% Ignore: modified every few iters.
hyperParams.showExamples = false; 
hyperParams.showConfusions = false;

Log(hyperParams.statlog, ['Model config: ' evalc('disp(hyperParams)')])
Log(hyperParams.statlog, ['Model training options: ' evalc('disp(options)')])
hyperParams = FlushLogs(hyperParams);

% TODO: Ressurect this feature or delete it.
hyperParams.firstSplit = 1;

% Trim out data files if needed
if hyperParams.datasetsPortion < 1
    p = randperm(length(splitFilenames));
    hyperParams.splitFilenames = hyperParams.splitFilenames(p...
        (1:round(hyperParams.datasetsPortion * length(hyperParams.splitFilenames))));
end

% Load saved parameters if available
savedParams = '';
if ~isempty(pretrainingFilename)
    savedParams = pretrainingFilename;
else
    listing = dir([options.name, '/ckpt-best*']);

    if ~isempty(listing)
        savedParams = [options.name, '/', listing(end).name];
    else 
        listing = dir([options.name, '/ckpt-*']);       
        if ~isempty(listing)
            savedParams = [options.name, '/', listing(end).name];
        end
    end
end
if ~isempty(savedParams)
    Log(hyperParams.statlog, ['Loading parameters: ' savedParams]);
    a = load(savedParams);
    modelState = a.modelState;
else
    modelState.step = 0;
    Log(hyperParams.statlog, ['Randomly initializing.']);
    [ modelState.theta, modelState.thetaDecoder, modelState.separateWordFeatures ] = ...
       InitializeModel(wordMap, hyperParams);
end

hyperParams = FlushLogs(hyperParams);

% Load training/test data
[trainDataset, testDatasets, hyperParams.trainingLengths] = ...
    LoadConstitDatasets(wordMap, relationMap, hyperParams);
% trainDataset = Symmetrize(trainDataset);

% Trim out individual examples if needed
if hyperParams.dataPortion < 1
    trainDataset = trainDataset(1:round(hyperParams.dataPortion * length(trainDataset)));
end

Log(hyperParams.statlog, 'Training')

if hyperParams.minFunc
    % Set up minFunc
    addpath('minFunc/minFunc/')
    addpath('minFunc/minFunc/compiled/')
    addpath('minFunc/minFunc/mex/')
    addpath('minFunc/autoDif/')

    % Warning: L-BFGS won't save state across restarts
    modelState.theta = minFunc(@ComputeFullCostAndGrad, modelState.theta, options, ...
        modelState.thetaDecoder, trainDataset, modelState.separateWordFeatures, hyperParams, 1);
else
    modelState.theta = TrainSGD(@ComputeFullCostAndGrad, modelState, options, ...
        trainDataset, hyperParams, testDatasets);
end
    
end
