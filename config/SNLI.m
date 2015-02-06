function [ hyperParams, options, wordMap, relationMap ] = SNLI(expName, dataflag, embDim, dim, topDepth, penult, lambda, tot, summing, mbs, lr, bottomDropout, topDropout, datamult, rtemult, nlimult, collo, tensorScale, wordScale, relu, dp)
% Configuration for experiments involving the SemEval SICK challenge and ImageFlickr 30k. 

[hyperParams, options] = Defaults();

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, '-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth),...
    '-pen', num2str(penult), '-lr', num2str(lr),...
    '-do', num2str(bottomDropout), '-', num2str(topDropout), '-co', num2str(collo),...
    '-m', num2str(datamult), '-tsc', num2str(tensorScale), '-wsc', num2str(wordScale),...
    '-mb', num2str(mbs), '-rte', num2str(rtemult), '-nli', num2str(nlimult),...
    '-dp', num2str(dp), '-relu', num2str(relu)];

if relu
  hyperParams.classNL = @LReLU;
  hyperParams.classNLDeriv = @LReLUDeriv;
end

hyperParams.useTrees = 0;
hyperParams.parensInSequences = 0;
hyperParams.lstm = 0;

hyperParams.dataPortion = dp;

% The dimensionality of the word/phrase vectors. Currently fixed at 25 to match
% the GloVe vectors.
hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;

% Used to compute the bound on the range for RNTN parameter initialization.
hyperParams.tensorScale = tensorScale; % 1?

% The raw range bound on word vectors.
hyperParams.wordScale = wordScale; % 0.1?

% Initialize the composition matrix with a fuzzy identity.
hyperParams.useEyes = 1;

if collo == 1
    hyperParams.vocabPath = ['../data/glove.6B.' num2str(embDim) 'd.txt'];
elseif collo == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The Collobert and Weston-sourced vectors only come in dim 50.'); 
elseif collo == 3
    hyperParams.vocabPath = ['../data/glove.840B.' num2str(embDim) 'd.txt'];
else
    hyperParams.vocabPath = ['../data/collo.scaled.' num2str(embDim) 'd.txt'];    
end

% The number of embedding transform layers. topDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.embeddingTransformDepth = 1;


% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = topDepth;

% If set, store embedding matrix gradients as spare matrices, and only apply regularization
% to the parameters that are in use at each step.
hyperParams.fastEmbed = true; % If we train words, go ahead and use it.

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = penult;

% Regularization coefficient.
hyperParams.lambda = lambda; % 0.002 works?;

% Apply dropout to the top feature vector of each tree, preserving activations
% with this probability. If this is set to 1, dropout is effectively not used.
hyperParams.bottomDropout = bottomDropout;
hyperParams.topDropout = topDropout;

% Use NTN layers in place of NN layers.
hyperParams.useThirdOrder = 0;
hyperParams.useThirdOrderComparison = 0;

hyperParams.useSumming = summing;

hyperParams.loadWords = true;
hyperParams.trainWords = true;

hyperParams.fragmentData = false;

% How many examples to run before taking a parameter update step on the accumulated gradients.
options.miniBatchSize = mbs;

options.updateFn = @AdaDeltaUpdate;

options.lr = lr;

if findstr(dataflag, 'sick-only')
    wordMap = InitializeMaps('sick_data/combined_words.txt');
    hyperParams.vocabName = 'sick_all'; 

    hyperParams.numRelations = 3;
    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}};
  relationMap = cell(1, 1);
  relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

    hyperParams.trainFilenames = {};    
    hyperParams.testFilenames = {};
  else
  wordMap = InitializeMaps('sick_data/combined_words.txt');
  hyperParams.vocabName = 'sick_all'; 

  hyperParams.numRelations = 3;
 	hyperParams.relations = {{'<', '|', '#'}};
relationMap = cell(1, 1);
relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

  hyperParams.trainFilenames = {};    
  hyperParams.testFilenames = {};
  hyperParams.splitFilenames = {'./SNLI/all-results-2014-2-1_b.txt'};
end

end
