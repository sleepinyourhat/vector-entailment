function [ hyperParams, options, wordMap, relationMap ] = Sick(expName, dataflag, embDim, dim, topDepth, penult, lambda, tot, summing, mbs, showgradmag, bottomDropout, topDropout, datamult, collo, parens, dp)
% Configuration for experiments involving the SemEval SICK challenge and ImageFlickr 30k. 

[hyperParams, options] = Defaults();

% The raw range bound on word vectors.
hyperParams.wordScale = 0.01;

% Used to compute the bound on the range for RNTN parameter initialization.
hyperParams.tensorScale = 1;

% Use an older initialization scheme for comparability with older experiments.
hyperParams.useCompatibilityInitialization = true;

% Generate an experiment name that includes all of the hyperparameter values that
% are being tuned.
hyperParams.name = [expName, 'COMPATINIT-', dataflag, '-l', num2str(lambda), '-dim', num2str(dim),...
    '-ed', num2str(embDim), '-td', num2str(topDepth),...
    '-pen', num2str(penult), '-sgm', num2str(showgradmag),...
    '-do', num2str(bottomDropout), '-', num2str(topDropout), '-co', num2str(collo),...
    '-m', num2str(datamult), '-par', num2str(parens),...
    '-mb', num2str(mbs), '-dp', num2str(dp), '-tot', num2str(tot), '-s', num2str(summing)];

if datamult < 0
  % Use the firstMultiplier method
  datamult = -1 * datamult;
  hyperParams.firstMultiplier = 30;
  hyperParams.firstCutoff = 2895;
end

hyperParams.classNL = @LReLU;
hyperParams.classNLDeriv = @LReLUDeriv;

hyperParams.dataPortion = dp;

% The dimensionality of the word/phrase vectors. Currently fixed at 25 to match
% the GloVe vectors.
hyperParams.dim = dim;
hyperParams.embeddingDim = embDim;

if collo == 1
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.6B.' num2str(embDim) 'd.txt'];
elseif collo == 2
    hyperParams.vocabPath = '/u/nlp/data/senna_embeddings/combined.txt';  
    assert(embDim == 50, 'The Collobert and Weston-sourced vectors only come in dim 50.'); 
elseif collo == 3
    hyperParams.vocabPath = ['/scr/nlp/data/glove_vecs/glove.840B.' num2str(embDim) 'd.txt'];
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

hyperParams.useEyes = 1;

if tot < 2
  hyperParams.useThirdOrder = tot;
  hyperParams.useThirdOrderComparison = tot;
elseif tot == 2
  hyperParams.lstm = 1;
  hyperParams.useTrees = 0;
  hyperParams.eyeScale = 0;
  hyperParams.useThirdOrder = 0;
  hyperParams.useThirdOrderComparison = 0;
  hyperParams.parensInSequences = 0;
elseif tot == 3
  hyperParams.lstm = 0;
  hyperParams.useTrees = 0;
  hyperParams.eyeScale = 0;
  hyperParams.useThirdOrder = 0;
  hyperParams.useThirdOrderComparison = 0;
  hyperParams.parensInSequences = 0;
end

hyperParams.useSumming = summing;

hyperParams.loadWords = true;
hyperParams.trainWords = true;

hyperParams.fragmentData = false;

% How many examples to run before taking a parameter update step on the accumulated gradients.
options.miniBatchSize = mbs;

options.updateFn = @AdaDeltaUpdate;

hyperParams.showGradMag = showgradmag;

if findstr(dataflag, 'sick-only-dev')
    wordMap = InitializeMaps('sick_data/combined_words.txt');
    hyperParams.vocabName = 'sick_all'; 

    hyperParams.numRelations = 3;
   	hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}};
	relationMap = cell(1, 1);
	relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt'};    
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
    				 './sick_data/SICK_trial_parsed_justneg.txt', ...
    				 './sick_data/SICK_trial_parsed_noneg.txt', ...
    				 './sick_data/SICK_trial_parsed_18plusparens.txt', ...
    				 './sick_data/SICK_trial_parsed_lt18_parens.txt'};
    hyperParams.splitFilenames = {};
elseif findstr(dataflag, 'sick-only')
    % The number of relations.
    hyperParams.numRelations = [3];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}};
    relationMap = cell(1, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

    wordMap = InitializeMaps('sick_data/combined_words.txt');
    hyperParams.vocabName = 'sick_all';

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed_exactAlign.txt', ...
                     './sick_data/SICK_train_parsed.txt', ...
                     './sick_data/SICK_trial_parsed_exactAlign.txt', ...
                     './sick_data/SICK_trial_parsed.txt'};
    hyperParams.testFilenames = {'./sick_data/SICK_test_annotated_rearranged_parsed_exactAlign.txt',...
                     './sick_data/SICK_test_annotated_rearranged_parsed.txt'};
    hyperParams.splitFilenames = {};
elseif strcmp(dataflag, 'sick-plus-600k-ea-dev') 
    % The number of relations.
    hyperParams.numRelations = [3 2];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'},
                             {'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(2, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
    relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));

    wordMap = InitializeMaps('sick_data/all_sick_plus_t4.txt');
    hyperParams.vocabName = 'aspt4';

    hyperParams.trainingMultipliers = [(datamult * 6); (datamult * 6); 1];

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed_exactAlign.txt', ...
                     './sick_data/SICK_train_parsed.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_600k.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed_exactAlign.txt', ...
                     './sick_data/SICK_trial_parsed.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_100.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 1, 2; 1, 1, 2; 0, 0, 0];
    hyperParams.testRelationIndices = [1, 1, 2];
elseif strcmp(dataflag, 'sick-plus-600k-dev') 
    % The number of relations.
    hyperParams.numRelations = [3 2];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'},
                             {'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(2, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
    relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));

    wordMap = InitializeMaps('sick_data/all_sick_plus_t4.txt');
    hyperParams.vocabName = 'aspt4';

    hyperParams.trainingMultipliers = [(datamult * 12); 1];

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_600k.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_trial_parsed.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_100.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 2; 1, 2; 0, 0];
    hyperParams.testRelationIndices = [1, 2];
elseif strcmp(dataflag, 'sick-plus-600k-ea') 
    % The number of relations.
    hyperParams.numRelations = [3 2];

    hyperParams.relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'},
                             {'ENTAILMENT', 'NONENTAILMENT'}};
    relationMap = cell(2, 1);
    relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));
    relationMap{2} = containers.Map(hyperParams.relations{2}, 1:length(hyperParams.relations{2}));

    wordMap = InitializeMaps('sick_data/all_sick_plus_t4.txt');
    hyperParams.vocabName = 'aspt4';

    hyperParams.trainingMultipliers = [(datamult * 6); (datamult * 6); (datamult * 6); (datamult * 6); 1];

    hyperParams.trainFilenames = {'./sick_data/SICK_train_parsed_exactAlign.txt', ...
                     './sick_data/SICK_train_parsed.txt', ...
                     './sick_data/SICK_trial_parsed_exactAlign.txt', ...
                     './sick_data/SICK_trial_parsed.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_600k.tsv'};
    hyperParams.testFilenames = {'./sick_data/SICK_test_annotated_rearranged_parsed_exactAlign.txt',...
                     './sick_data/SICK_test_annotated_rearranged_parsed.txt', ...
                     './sick_data/SICK_trial_parsed_exactAlign.txt', ...
                     '/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_100.tsv'};
    hyperParams.splitFilenames = {};
    % Use different classifiers for the different data sources.
    hyperParams.relationIndices = [1, 1, 1, 1, 2; 1, 1, 1, 2, 0; 0, 0, 0, 0, 0];
    hyperParams.testRelationIndices = [1, 1, 1, 2];
elseif strcmp(dataflag, 'imageflickrshort')
    % The number of relations.
    hyperParams.numRelations = 2; 

    hyperParams.relations = {{'ENTAILMENT', 'NONENTAILMENT'}};
	relationMap = cell(1, 1);
	relationMap{1} = containers.Map(hyperParams.relations{1}, 1:length(hyperParams.relations{1}));

    wordMap = InitializeMaps('sick_data/flickr_words_t4.txt');
    hyperParams.vocabName = 'spt4-2cl';

    hyperParams.splitFilenames = {'/scr/nlp/data/ImageFlickrEntailments/shuffled_clean_parsed_entailment_pairs_600k.tsv'};
    hyperParams.testFilenames = {'/scr/nlp/data/ImageFlickrEntailments/clean_parsed_entailment_pairs_first500.tsv', ...
    				 './sick_data/clean_parsed_entailment_pairs_second10k_first500.tsv'};
    hyperParams.trainFilenames = {};
end

end
