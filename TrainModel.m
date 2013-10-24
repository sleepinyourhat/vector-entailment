function TrainModel(dim, lambda, pretrainingFilename, testFilenames, splitFilenames)

[worddata, wordMap, relationMap, relations] = ...
    LoadTrainingData('wordpairs-v2.tsv');

% Set up minfunc
addpath('minFunc/minFunc/')
addpath('minFunc/minFunc/compiled/')
addpath('minFunc/minFunc/mex/')
addpath('minFunc/autoDif/')

% Set up hyperparameters:
hyperParams.dim = dim;
hyperParams.numRelations = 7;
hyperParams.penultDim = 21;
hyperParams.lambda = lambda;
hyperParams.relations = relations;
hyperParams.minFunc = false;

disp(hyperParams)

[ theta, thetaDecoder ] = InitializeModel(size(wordMap, 1), hyperParams);

% minfunc options (not tuned)
options.Method = 'lbfgs';
options.MaxFunEvals = 300;
options.DerivativeCheck = 'on';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];

% adaGradSGD options (not tuned)
options.numPasses = 600;
options.miniBatchSize = 32;
options.lr = 0.01;

% Add relation vector, so it can be ref'd in error reporting.
disp(options)

if nargin > 3 && ~isempty(pretrainingFilename)
    clear 'theta'
    clear 'thetaDecoder'
    v = load(pretrainingFilename);
    theta = v.theta;
    thetaDecoder = v.thetaDecoder;
else
    % Pretrain words
    disp('Pretraining')
    if hyperParams.minFunc
        theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, worddata, hyperParams);
        theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, worddata, hyperParams);
        theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, worddata, hyperParams);
        theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, worddata, hyperParams);
        % Forget and repeat?
    else
        theta = adaGradSGD(theta, options, thetaDecoder, worddata, hyperParams);
    end

    save(['pretrained-theta-wordpairs-', num2str(hyperParams.dim), 'x', num2str(hyperParams.penultDim)],...
      'theta', 'thetaDecoder')
end
  
% Evalaute on word pair data
[~, ~, preAcc, preConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, worddata, hyperParams);


disp('Word pair confusion, PER: ')
disp('tr:  #     =     >     <     |     ^     v')
disp(preConfusion)
disp(preAcc)

% Reset composition function for training
theta = ReinitializeCompositionLayer (theta, thetaDecoder, hyperParams);


% Load training data
allConstitFilenames = {'BQ-MTI-all-some.tsv', ...
'BQ-MTI-no-all.tsv',		'MTI-MTI-Thai-animal.tsv', ...
'BQ-MTI-no-some.tsv',		'MTI-Thai-null.tsv', ...
'BQ-all-no.tsv',			'MTI-null-Thai.tsv', ...
'BQ-all-some.tsv',			'NBQ-most-two.tsv', ...
'BQ-some-no.tsv',			'NBQ-three-most.tsv', ...
'MQ-all-most.tsv',			'NBQ-three-two.tsv', ...
'MQ-all-three.tsv',         'NEG-MTI-not-Thai-null.tsv', ...
'MQ-all-two.tsv',			'NEG-notnot-null.tsv', ...
'MT-animal.tsv',            'BQ-all-no-able.tsv', ...
'BQ-some-no-able.tsv',      'BQ-all-some-able.tsv', ...
'BQ-some-no-Thai.tsv',      'BQ-MTI-all-some-able.tsv', ...
'BQ-MTI-no-all-able.tsv',   'BQ-MTI-no-some-able.tsv'}
trainFilenames = allConstitFilenames;
testInd=find(ismember(allConstitFilenames,testFilenames));
trainFilenames(testInd) = []
[trainDataset, testDatasets] = ...
    LoadConstitDatasets(trainFilenames, splitFilenames, testFilenames, wordMap, relationMap)

% Train
disp('Training')
options.MaxFunEvals = 300;
options.DerivativeCheck = 'off';

if hyperParams.minFunc
    theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, trainDataset, hyperParams);
    % Forget and repeat?
else
    theta = adaGradSGD(theta, options, thetaDecoder, trainDataset, hyperParams);
end

% Evaluate on training data
[~, ~, trAcc, trConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, trainDataset, hyperParams);

disp('Training confusion, PER: ')
disp('tr:  #     =     >     <     |     ^     v')
disp(trConfusion)
disp(trAcc)

% Evaluate on test datasets, and show set-by-set results
datasetNames = [testFilenames, splitFilenames];
aggConfusion = zeros(hyperParams.numRelations);
for i = 1:length(testDatasets)
    [~, ~, acc, confusion] = ComputeFullCostAndGrad(theta, thetaDecoder, testDatasets{i}, hyperParams);
    disp(['Test confusion, PER for ', datasetNames{i}, ':'])
    disp('tr:  #     =     >     <     |     ^     v')
    disp(confusion)
    disp(acc) 
    aggConfusion = aggConfusion + confusion;
end

% Compute error rate from summed confusion matrix
aggAcc = 1 - sum(sum(eye(hyperParams.numRelations) .* aggConfusion)) / sum(sum(aggConfusion));    

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
disp(aggConfusion)
disp(aggAcc)

end
