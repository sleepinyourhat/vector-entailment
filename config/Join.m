function [ hyperParams, options, wordMap, labelMap ] = Join(name, dataflag, dim, penult, top, lambda, tot, relu, tdrop, mbs)
% Label composition experiments.
% See Defaults.m for parameter descriptions.

[hyperParams, options] = Defaults();

hyperParams.name = [name, '-', dataflag, '-d', num2str(dim), '-pen', num2str(penult), '-top', num2str(top), ...
				    '-tot', num2str(tot), '-relu', num2str(relu), '-l', num2str(lambda),...
				    '-dropout', num2str(tdrop), '-mb', num2str(mbs)];

if tot == -1
  hyperParams.useTrees = 0;
  hyperParams.useThirdOrderComposition = 0;
  hyperParams.useThirdOrderMerge = 0;
  hyperParams.useSumming = 1;
else
  % Optionally use NTN layers in place of NN layers.
  hyperParams.useThirdOrderComposition = tot;
  hyperParams.useThirdOrderMerge = tot;
end

hyperParams.dim = dim;
hyperParams.embeddingDim = dim;
hyperParams.penultDim = penult;
hyperParams.lambda = lambda; % 0.002 works?;
hyperParams.topDepth = top;
hyperParams.topDropout = tdrop;

hyperParams.vocabName = dataflag;

options.miniBatchSize = mbs;

hyperParams.labels = {{'#', '=', '>', '<', '|', '^', 'v'}};
hyperParams.numLabels = [7];
labelMap = cell(1, 1);
labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));


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

end
