function HoldOneOut(filename, expName)
% Run hold-one-out experiments. 
disp('MAY BE OUT OF DATE. DO NOT TRUST.');

[~, wordMap, relationMap, relations] = ...
    LoadTrainingData('wordpairs-v2.tsv');

% disp('Uninformativizing:');
% worddata = Uninformativize(worddata);

% Set up hyperparameters:
hyperParams.dim = 16;
hyperParams.numRelations = 7; 
hyperParams.topDepth = 1;
hyperParams.penultDim = 21;
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
hyperParams.classNL = @LReLU;
hyperParams.classNLDeriv = @LReLUDeriv;

disp(hyperParams)

% adaGradSGD options (partially tuned)
options.numPasses = 100;
options.miniBatchSize = 32; % tuned-ish
options.lr = 0.05;

% adaGradSGD display options
options.testFreq = 4;
options.confusionFreq = 32; % should be a multiple of testfreq
options.examplesFreq = 32; % should be a multiple of testfreq
options.checkpointFreq = 1000;
options.name = expName;
options.runName = 'pre';
mkdir(options.name); 

% Add relation vector, so it can be ref'd in error reporting.
disp(options)

trainFilenames = {filename};

% allConstitFilenames = [allConstitFilenames, beyondQ];

[trainDataset, ~] = ...
    LoadConstitDatasets(trainFilenames, {}, {}, wordMap, relationMap);

aggTrConfusion = zeros(hyperParams.numRelations);
aggTeConfusion = zeros(hyperParams.numRelations);

for i = 1:length(trainDataset)
    testDataset = trainDataset(i);
    trimmedTrainDataset = trainDataset;
    trimmedTrainDataset(i) = [];
    symTrainDataset = Symmetrize(trimmedTrainDataset);
    
    % Train
    disp('Training')
    options.runName = num2str(i);
    [ theta, thetaDecoder ] = InitializeModel(size(wordMap, 1), hyperParams);

    theta = adaGradSGD(theta, options, thetaDecoder, symTrainDataset, hyperParams);

    % Evaluate on training data
    [~, ~, ~, trConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, symTrainDataset, hyperParams);
    aggTrConfusion = aggTrConfusion + trConfusion;

    % Evaluate on test minidataset
    hyperParams.quiet = false;
    [~, ~, acc, confusion] = ComputeFullCostAndGrad(theta, thetaDecoder, testDataset, hyperParams);
    hyperParams.quiet = true;
    aggTeConfusion = aggTeConfusion + confusion;
    disp(['Test PER for ', num2str(i), ':'])
    disp(acc) 
end

% Compute error rate from summed confusion matrix
aggTeAcc = 1 - sum(sum(eye(hyperParams.numRelations) .* aggTeConfusion)) / sum(sum(aggTeConfusion));    
aggTrAcc = 1 - sum(sum(eye(hyperParams.numRelations) .* aggTrConfusion)) / sum(sum(aggTrConfusion));    


% Print results for all three full datasets
disp('Training confusion, PER: ')
disp('tr:  #     =     >     <     |     ^     v')
disp(aggTrConfusion)
disp(aggTrAcc)

disp('Test confusion, PER: ')
disp('tr:  #     =     >     <     |     ^     v')
disp(aggTeConfusion)
disp(aggTeAcc)

end
