function [ hyperParams, options, wordMap, labelMap ] = GradCheck(transDepth, topDepth, composition, trainwords, fastemb, multipleClassSets, sentiment)
% Set up a gradient check for the main learned parameters.

% TrainModel('', 1, @GradCheck, 1, 2, 4, 1, 0, 1, 0);
% NOTE: The LatticeLSTM doesn't do especially well on gradient checks at random initialization
% but basically passes if tested on the model state after a few steps... keep an eye on this.

[hyperParams, options] = Defaults();

hyperParams = CompositionSetup(hyperParams, composition);


    hyperParams.compNL = @ReLU;
    hyperParams.compNLDeriv = @ReLUDeriv; 
    hyperParams.classNL = @ReLU;
    hyperParams.classNLDeriv = @ReLUDeriv;

hyperParams.name = 'gradcheck';

% The dimensionality of the word/phrase vectors.
hyperParams.dim = 3;
hyperParams.embeddingDim = 3;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = 3;

hyperParams.testFraction = 0.33;

hyperParams.parensInSequences = false;

% Test text file loading, rather than importing preprocessed data.
hyperParams.ignorePreprocessedFiles = true;

% The number of embedding transform layers. transDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.embeddingTransformDepth = transDepth;

hyperParams.topDepth = topDepth;

% If set, store embedding matrix gradients as spare matrices, and only apply regularization
% to the parameters that are in use at each step.
hyperParams.fastEmbed = fastemb;

% Turn off gradient/delta clipping, since that distrorts gradients.
hyperParams.clipGradients = false;
hyperParams.maxGradNorm = inf;
hyperParams.maxDeltaNorm = inf;

% If set, weight the supervision higher in the lattice by the product of the probabilities 
% of the correct merge positions lower in the lattice to avoid training the composition model on bad inputs.
% This is defined in a way that is guaranteed to cause the gradient check to fail.
hyperParams.latticeLocalCurriculum = false;

% Turn off regularization to more clearly show patterns in the main gradinets.
hyperParams.lambda = 0;

% Hack: -1 => always drop out the same one unit.
hyperParams.bottomDropout = -1;
hyperParams.topDropout = -1;

hyperParams.loadWords = false;
hyperParams.trainWords = trainwords;

hyperParams.minFunc = true;
options.DerivativeCheck = 'on';

wordMap = InitializeMaps('./quantifiers/wordlist.tsv'); 
hyperParams.vocabName = 'quantifiers';

if sentiment 
	hyperParams.sentenceClassificationMode = 1;
	hyperParams.SSTMode = 1;
	wordMap = InitializeMaps('./sst-data/gradcheckwords.txt');
	hyperParams.vocabName = 'sst-gc'; 

	hyperParams.numLabels = [5];
	hyperParams.labelCostMultipliers = [1 0.0001 3 0.25 1];

	hyperParams.labels = {{'0', '1', '2', '3', '4'}};
	labelMap = cell(1, 1);
	labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

	hyperParams.trainFilenames = {};    
	hyperParams.splitFilenames = {'./sst-data/gradcheckfile.txt'};    
	hyperParams.testFilenames = {};
else
	if ~multipleClassSets
		hyperParams.labels = {{'#', '=', '>', '<', '|', '^', 'v'}};
		hyperParams.labelCostMultipliers = [1 0.0001 3 0.25 1 0.5 0.5];
		hyperParams.numLabels = [7];
		labelMap = cell(1, 1);
		labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));

		hyperParams.splitFilenames = {'./quantifiers/test_file.tsv'};
		hyperParams.trainFilenames = {};
		hyperParams.testFilenames = {};
	else
		hyperParams.labels = {{'#', '=', '>', '<', '|', '^', 'v'},
								 {'a', 'b', 'c', '#', '=', '>', '<', '|', '^', 'v'}};
		hyperParams.numLabels = [7, 10];
		hyperParams.labelIndices = [0, 0; 0, 0; 1, 2];
	    hyperParams.testLabelIndices = [1, 2];

		labelMap = cell(2, 1);
		labelMap{1} = containers.Map(hyperParams.labels{1}, 1:length(hyperParams.labels{1}));
		labelMap{2} = containers.Map(hyperParams.labels{2}, 1:length(hyperParams.labels{2}));

		hyperParams.splitFilenames = {'./quantifiers/test_file.tsv', './quantifiers/test_file_2.tsv'};
		hyperParams.trainFilenames = {};
		hyperParams.testFilenames = {};
	end
end

end
