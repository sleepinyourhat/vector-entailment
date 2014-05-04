% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function HoldOneOutJoin(expName, trainingDataFile)
% Run hold-one-out experiments. 
disp('MAY BE OUT OF DATE. DO NOT TRUST.');

[ wordMap, ~, relations, trainDataset] = ...
    LoadTrainingData(trainingDataFile);

trainDataset
% trainDataset = Asymmetrize(trainDataset)

% disp('Uninformativizing:');
% worddata = Uninformativize(worddata);

% Set up hyperparameters:
hyperParams.dim = 11;
hyperParams.numRelations = 7; 
hyperParams.topDepth = 1;
hyperParams.penultDim = 45;
hyperParams.lambda = 0.00002;
hyperParams.relations = relations;
hyperParams.noPretraining = true;
hyperParams.minFunc = false;
hyperParams.showExamples = false;
hyperParams.showConfusions = false;
hyperParams.norm = 2;
hyperParams.untied = false;
hyperParams.firstSplit = 1;
hyperParams.useThirdOrder = 1;

% Nonlinearities
hyperParams.compNL = @Sigmoid;
hyperParams.compNLDeriv = @SigmoidDeriv; 
hyperParams.classNL = @LReLU;
hyperParams.classNLDeriv = @LReLUDeriv;

disp(hyperParams)

% adaGradSGD options (partially tuned)
options.numPasses = 1000;
options.miniBatchSize = 32; % tuned-ish
options.lr = 0.2;
options.resetSumSqFreq = 1000;

% adaGradSGD display options
options.testFreq = 1;
options.confusionFreq = 1000; % should be a multiple of testfreq
options.examplesFreq = 1000; % should be a multiple of testfreq
options.checkpointFreq = 1000; % Don't bother checkpointing here.
options.name = expName;
options.runName = 'pre';
options.Method = 'lbfgs';
options.MaxFunEvals = 1000;
options.DerivativeCheck = 'off';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
options.OutputFcn = @Display;
mkdir(options.name); 

% Add relation vector, so it can be ref'd in error reporting.
disp(options)

% trainFilenames = {filename};

% allConstitFilenames = [allConstitFilenames, beyondQ];

% [trainDataset, ~] = ...
%    LoadConstitDatasets(trainFilenames, {}, {}, wordMap, relationMap);

aggTrConfusion = zeros(hyperParams.numRelations);
aggTeConfusion = zeros(hyperParams.numRelations);

order = randperm(length(trainDataset))
for j = 1:length(trainDataset)
    i = j;
    testDataset = trainDataset(i);
    trimmedTrainDataset = trainDataset;
    trimmedTrainDataset(i) = [];
    symTrainDataset = Symmetrize(trimmedTrainDataset);
    
    % Train
    disp('Training')
    options.runName = num2str(i);
    [ theta, thetaDecoder ] = InitializeModel(size(wordMap, 1), hyperParams);

    if ~hyperParams.minFunc
        theta = AdaGradSGD(theta, options, thetaDecoder, symTrainDataset, hyperParams, {{[testDataset.leftTree.getText(), testDataset.rightTree.getText()]},{testDataset}});
    else
        addpath('minFunc/minFunc/')
        addpath('minFunc/minFunc/compiled/')
        addpath('minFunc/minFunc/mex/')
        addpath('minFunc/autoDif/')

        theta = minFunc(@ComputeFullCostAndGrad, theta, options, ...
            thetaDecoder, symTrainDataset, hyperParams, {{[testDataset.leftTree.getText(), testDataset.rightTree.getText()]},{testDataset}});
    end
        
    % Evaluate on training data
    [~, ~, ~, trConfusion] = ComputeFullCostAndGrad(theta, thetaDecoder, symTrainDataset, hyperParams);
    aggTrConfusion = aggTrConfusion + trConfusion;

    % Evaluate on test minidataset
    hyperParams.quiet = false;
    [~, ~, acc, confusion] = ComputeFullCostAndGrad(theta, thetaDecoder, testDataset, hyperParams);
    hyperParams.quiet = true;
    aggTeConfusion = aggTeConfusion + confusion;
    disp(['Test PER for ', num2str(i), '-', testDataset.leftTree.getText(), ' ', ...
                  hyperParams.relations{testDataset.relation}, ' ', ... 
                  testDataset.rightTree.getText, ':'])

    disp(acc) 
    
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
