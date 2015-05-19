% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function TrainModel(pretrainingFilename, fold, ConfigFn, varargin)
% The main training and testing script.

% Arguments:
%% pretrainingFilename: Pass in the filename to a checkpoint file to start training
%%%% from that checkpoint with reset optimization. 
%%%% If this argument is blank but a checkpoint with the exact same
%%%% experiment name is found, that checkpoint will be loaded. To start a fresh experiment,
%%%% use a fresh experiment name (set in the config file).
%% fold: Used for five fold cross-validation on some data sources. Settings other than 1 will 
%%%% cause -f# to be appended to the experiment name.
%% ConfigFn: A function handle to an experiment configuration function that sets up 
%%%% hyperParams, options, wordMap, and labelMap. See examples in the config/ directory.
%% varargin: All remaining arguments will be passed through to the config function.

% Look for experiment configuration scripts in the config/ directory.
addpath('config/')

% Look for the NN internals in this directory.
addpath('layer-fns/')

% Set up paralellization
if isempty(gcp('nocreate'))
    c = parcluster();
    t = tempname();
    mkdir(t);
    c.JobStorageLocation = t;
    c.NumWorkers = 4;
    if exist('parpool')
      % >= 2013b
      parpool(c, 4);
    else
      % < 2013b
      matlabpool(c, 4);
    end
end

[ hyperParams, options, wordMap, labelMap ] = ConfigFn(varargin{:});
hyperParams.labelRanges = ComputeLabelRanges(labelMap);

% If the fold number is grater than one, the train/test split on split data will 
% be offset accordingly.
hyperParams.foldNumber = fold;

% The name assigned to the current full run. Used in checkpoint naming, and must
% match the directory created above.
if fold > 1
    hyperParams.name = [hyperParams.name, '-f', num2str(fold)];
end
options.name = hyperParams.name;

% Set up an experiment directory.
mkdir(hyperParams.name); 

hyperParams.statlog = fopen([hyperParams.name '/stat_log'], 'a');
hyperParams.examplelog = fopen([hyperParams.name '/example_log'], 'a');

% Ignore: modified every few iters.
hyperParams.showExamples = false; 
hyperParams.showDetailedStats = true;

Log(hyperParams.statlog, ['hyperParams: \n' evalc('disp(hyperParams)')])
Log(hyperParams.statlog, ['options: \n' evalc('disp(options)')])
hyperParams = FlushLogs(hyperParams);

% TODO: Ressurect this feature or delete it.
hyperParams.firstSplit = 1;

% Trim out data files if needed
if hyperParams.datasetsPortion < 1
    p = randperm(length(splitFilenames));
    hyperParams.splitFilenames = hyperParams.splitFilenames(p...
        (1:round(hyperParams.datasetsPortion * length(hyperParams.splitFilenames))));
end

% Set up GPU, borrowed from Thang Luong
if hyperParams.gpu
    n = gpuDeviceCount;  
    if n>0 % GPU exists
        Log(hyperParams.statlog, ['Using GPU 1.'])
        gpuDevice(1);
    else
        Log(hyperParams.statlog, ['No GPU found. Using CPUs.'])
        hyperParams.gpu = false;
    end    
end

% Look for previously saved checkpoint files.
savedParams = '';
if ~isempty(pretrainingFilename)
    modelState.step = 0;

    Log(hyperParams.statlog, ['Randomly initializing.']);
    [ modelState.theta, modelState.thetaDecoder, modelState.separateWordFeatures ] = ...
       InitializeModel(wordMap, hyperParams);

    savedParams = pretrainingFilename;

    Log(hyperParams.statlog, ['Loading transfer parameters: ' savedParams]);
    a = load(savedParams);

    modelState = TransferInitialization(modelState, a.modelState, wordMap, hyperParams.sourceWordMap);
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

    % Load the checkpoint if present.
    if ~isempty(savedParams)
        Log(hyperParams.statlog, ['Loading checkpoint: ' savedParams]);
        a = load(savedParams);
        modelState = a.modelState;
    else
        modelState.step = 0;
        Log(hyperParams.statlog, ['Randomly initializing.']);
        [ modelState.theta, modelState.thetaDecoder, modelState.separateWordFeatures ] = ...
           InitializeModel(wordMap, hyperParams);
    end
end


% Set up the special word indices for the lattice model
if hyperParams.useLattices
    % These tokens should not occur in running text, but are instead meant to indicade edges for the lattice scorer.
    hyperParams.sentenceStartWordIndex = wordMap('<s>');
    hyperParams.sentenceEndWordIndex = wordMap('</s>');
end

hyperParams = FlushLogs(hyperParams);

% Load training/test data
[ trainDataset, testDatasets, hyperParams.trainingLengths ] = ...
    LoadAllDatasets(wordMap, labelMap, hyperParams);

% Trim out individual examples if needed (only from the first source)
if hyperParams.dataPortion < 1
    Log(hyperParams.statlog, 'Trimming first training data set.')
    trainDataset = trainDataset([1:round(hyperParams.dataPortion * hyperParams.trainingLengths(1)),...
                                 (round(hyperParams.trainingLengths(1)) + 1):length(trainDataset)]);
    hyperParams.trainingLengths(1) = round(hyperParams.dataPortion * hyperParams.trainingLengths(1));
end

Log(hyperParams.statlog, 'Training')

% Choose a function of the data to optimize.
if ~hyperParams.sentenceClassificationMode && ~hyperParams.useTrees
    % Entailment
    optFn = @ComputeBatchEntailmentCostAndGrad;
elseif ~hyperParams.useTrees
    % Sentiment/classification
    optFn = @ComputeBatchSentenceClassificationCostAndGrad;
else
    % Trees in either mode
    optFn = @ComputeUnbatchedCostAndGrad;
end

% Launch the optimizer!
if hyperParams.minFunc
    % Use minFunc -- only suitable for gradient checking and small toy problems.

    % Set up minFunc
    addpath('minFunc/minFunc/')
    addpath('minFunc/minFunc/compiled/')
    addpath('minFunc/minFunc/mex/')
    addpath('minFunc/autoDif/')

    modelState.theta = minFunc(optFn, modelState.theta, options, ...
        modelState.thetaDecoder, trainDataset, modelState.separateWordFeatures, hyperParams, 1);
else
    modelState.theta = TrainSGD(optFn, modelState, options, trainDataset, hyperParams, testDatasets);
end
    
end
