% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainModel(pretrainingFilename, expName, fold, ConfigFn, varargin)
% The main training and testing script. The first arguments to the function
% have been tweaked quite a few times depending on what is being tuned.

% Look for configuration scripts in the config/ directory.
addpath('config/')

[ hyperParams, options, wordMap, relationMap ] = ConfigFn(varargin{:});

% If the fold number is grater than one, the train/test split on split data will 
% be offset accordingly.
hyperParams.foldNumber = fold;

% The name assigned to the current full run. Used in checkpoint naming, and must
% match the directory created above.
hyperParams.name = expName;
options.name = expName;

% Set up an experimental directory.
if nargin > 4
    mkdir(hyperParams.name); 
else
    expName = '.';
end

hyperParams.statlog = fopen([hyperParams.name '/stat_log'], 'a');
hyperParams.examplelog = fopen([hyperParams.name '/example_log'], 'a');

% Ignore: modified every few iters.
hyperParams.showExamples = false; 
hyperParams.showConfusions = false;

Log(hyperParams.statlog, ['Model config: ' evalc('disp(hyperParams)')])
Log(hyperParams.statlog, ['Model training options: ' evalc('disp(options)')])

% Remove the test data from the split data
hyperParams.splitFilenames = setdiff(hyperParams.splitFilenames, hyperParams.testFilenames);

% TODO
hyperParams.firstSplit = length(hyperParams.testFilenames) + 1;

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
    listing = dir([options.name, '/', 'ckpt-tr@*']);
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
    [ modelState.theta, modelState.thetaDecoder, modelState.constWordFeatures ] = ...
       InitializeModel(wordMap, hyperParams);
end

% Load training/test data
[trainDataset, testDatasets] = ...
    LoadConstitDatasets(wordMap, relationMap, hyperParams);
% trainDataset = Symmetrize(trainDataset);

% Trim out individual examples if needed
if hyperParams.dataPortion < 1
    trainDataset = trainDataset(1:round(hyperParams.dataPortion * length(trainDataset)));
end

% Train
Log(hyperParams.statlog, 'Training')

if hyperParams.minFunc
    % Set up minFunc
    addpath('minFunc/minFunc/')
    addpath('minFunc/minFunc/compiled/')
    addpath('minFunc/minFunc/mex/')
    addpath('minFunc/autoDif/')

    % Warning: L-BFGS won't save state across restarts
    modelState.theta = minFunc(@ComputeFullCostAndGrad, modelState.theta, options, ...
        modelState.thetaDecoder, trainDataset, modelState.constWordFeatures, hyperParams, testDatasets);
else
    modelState.theta = AdaGradSGD(@ComputeFullCostAndGrad, modelState, options, ...
        trainDataset, hyperParams, testDatasets);
end
    
end
