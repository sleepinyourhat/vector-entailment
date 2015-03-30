function [ hyperParams, options, wordMap, relationMap ] = Join(name, dataflag, dim, penult, top, lambda, tot, relu, tdrop, mbs)
% Relation composition experiments.

[hyperParams, options] = Defaults();

hyperParams.name = [name, '-', dataflag, '-d', num2str(dim), '-pen', num2str(penult), '-top', num2str(top), ...
				    '-tot', num2str(tot), '-relu', num2str(relu), '-l', num2str(lambda),...
				    '-dropout', num2str(tdrop), '-mb', num2str(mbs)];


% The raw range bound on word vectors.
hyperParams.wordScale = 0.01;

% Used to compute the bound on the range for RNTN parameter initialization.
hyperParams.tensorScale = 1;

% Use an older initialization scheme for comparability with older experiments.
hyperParams.useCompatibilityInitialization = true;

hyperParams.useEyes = 1;

if tot == -1
  hyperParams.useThirdOrder = 0;
  hyperParams.useThirdOrderComparison = 0;
  hyperParams.useSumming = 1;
else
  % Use NTN layers in place of NN layers.
  hyperParams.useThirdOrder = tot;
  hyperParams.useThirdOrderComparison = tot;
end

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

hyperParams.topDepth = top;

hyperParams.topDropout = tdrop;

if relu
  hyperParams.classNL = @LReLU;
  hyperParams.classNLDeriv = @LReLUDeriv;
end

hyperParams.vocabName = dataflag;

hyperParams.relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numRelations = [7];
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));


if findstr(dataflag, 'fold1')
    % Choose which files to load in each category.
    hyperParams.splitFilenames = {};
    hyperParams.trainFilenames = {'./join-table/data/train_1.tsv'};
    hyperParams.testFilenames = {'./join-table/data/test_1.tsv', ...
                     './join-table/data/underivable_1.tsv'};
    wordMap = LoadWordPairData('./join-table/data/train_1.tsv')
elseif findstr(dataflag, 'fold2')
    % Choose which files to load in each category.
    hyperParams.splitFilenames = {};
    hyperParams.trainFilenames = {'./join-table/data/train_2.tsv'};
    hyperParams.testFilenames = {'./join-table/data/test_2.tsv', ...
                     './join-table/data/underivable_2.tsv'};
    wordMap = LoadWordPairData('./join-table/data/train_2.tsv')
elseif findstr(dataflag, 'fold3')
    % Choose which files to load in each category.
    hyperParams.splitFilenames = {};
    hyperParams.trainFilenames = {'./join-table/data/train_3.tsv'};
    hyperParams.testFilenames = {'./join-table/data/test_3.tsv', ...
                     './join-table/data/underivable_3.tsv'};
    wordMap = LoadWordPairData('./join-table/data/train_3.tsv')
elseif findstr(dataflag, 'fold4')
    % Choose which files to load in each category.
    hyperParams.splitFilenames = {};
    hyperParams.trainFilenames = {'./join-table/data/train_4.tsv'};
    hyperParams.testFilenames = {'./join-table/data/test_4.tsv', ...
                     './join-table/data/underivable_4.tsv'};
    wordMap = LoadWordPairData('./join-table/data/train_4.tsv')
elseif findstr(dataflag, 'fold5')
    % Choose which files to load in each category.
    hyperParams.splitFilenames = {};
    hyperParams.trainFilenames = {'./join-table/data/train_5.tsv'};
    hyperParams.testFilenames = {'./join-table/data/test_5.tsv', ...
                     './join-table/data/underivable_5.tsv'};
    wordMap = LoadWordPairData('./join-table/data/train_5.tsv');
else
    % Choose which files to load in each category.
    hyperParams.splitFilenames = {};
    hyperParams.trainFilenames = {'./join-table/data/6x80_train.tsv'};
    hyperParams.testFilenames = {'./join-table/data/6x80_test.tsv', ...
                     './join-table/data/6x80_test_underivable.tsv'};
    wordMap = LoadWordPairData('./join-table/data/6x80_train.tsv');
end

options.numPasses = 2600;

options.miniBatchSize = mbs;


end
