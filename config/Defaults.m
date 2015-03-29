function [ hyperParams, options ] = Defaults()
% Set defaults for all model and learning parameters.

hyperParams.name = ['rnn' datestr(now, 'yymmddHHMMSS')];

% Use Tree or Pyramid to represent data if set. Else, use Sequence.
hyperParams.useTrees = 1;
hyperParams.usePyramids = 0;

% If set, and if useTrees is false, use an LSTM RNN.
hyperParams.lstm = 0;

% The dimensionality of the word/phrase vectors.
hyperParams.dim = 25;
hyperParams.embeddingDim = 25;

% How much of a contribution (in the range 0-1) should tensors give to outputs at initialization.
hyperParams.tensorScale = 0.9;

% How much of the output of the matrix parameters (in the range 0-1) should be initialized with an identity matrix.
hyperParams.eyeScale = 0.5;

% Which initialization scheme to use
hyperParams.NNinitType = 1;
hyperParams.NTNinitType = 1;
hyperParams.LSTMinitType = 2;

% Use an older initialization scheme for comparability with older experiments.
hyperParams.useCompatibilityInitialization = false;

% How far *in each direction* should the connection classifier in a Pyramid model look.
% Setting this to 1 means to only use the immediate left and right composition inputs with no
% additional context.
hyperParams.pyramidConnectionContextWidth = 3;

% The number of embedding transform layers. topDepth > 0 means NN layers will be
% added above the embedding matrix. This is likely to only be useful when
% learnWords is false, and so the embeddings do not exist in the same space
% the rest of the constituents do.
hyperParams.embeddingTransformDepth = 0;

% The number of comparison layers. topDepth > 1 means NN layers will be
% added between the RNTN composition layer and the softmax layer.
hyperParams.topDepth = 1;

% The dimensionality of the comparison layer(s).
hyperParams.penultDim = 75;

% Regularization coefficient.
hyperParams.lambda = 0;

% Apply dropout to the top feature vector of each tree, preserving activations
% with this probability. If this is set to 1, dropout is effectively not used.
hyperParams.bottomDropout = 1;
hyperParams.topDropout = 1;

% L1 v. L2 regularization. If no regularization is needed, set
% lambda to 0 and ignore this parameter.
hyperParams.norm = 2;

% Use the syntactically untied composition layer params.
% NOTE: This is not well supported right now.
hyperParams.untied = false; 

% Use only the specified fraction of the training datasets
hyperParams.datasetsPortion = 1.0;

% Use only the specified fraction of the individual training examples
hyperParams.dataPortion = 1.0;

% When a dataset is to be automatically split into train and test portions,
% use this much as test data.
hyperParams.testFraction = 0.2;

% When evaluating random samples from a training data set, don't evaluate
% more than this many in each session.
hyperParams.maxTrainingEvalSampleSize = 1000;

% Use NTN layers in place of NN layers.
hyperParams.useThirdOrder = true;
hyperParams.useThirdOrderComparison = true;

% Use a simple summing layer in place of the composition (R)NN layer.
% useThirdOrder should be false if this is used.
hyperParams.useSumming = false;

% If set, train using minFunc. Only partially supported. See GradCheck for an example.
hyperParams.minFunc = false;

hyperParams.loadWords = false;
hyperParams.trainWords = true;
hyperParams.vocabPath = '';

% Nonlinearities.
hyperParams.compNL = @TanhActivation;
hyperParams.compNLDeriv = @TanhDeriv; 
hyperParams.classNL = @TanhActivation;
hyperParams.classNLDeriv = @TanhDeriv;

% If set, don't try to keep the entire training data set in memory at once.
hyperParams.fragmentData = false;

% If set, store embedding matrix gradients as spare matrices, and only apply regularization
% to the parameters that are in use at each step. This does nothing if trainWords is false.
% Useful as long as the vocabulary size is fairly large. (Exact threshold unknown.)
hyperParams.fastEmbed = false;

hyperParams.clearActivations = false;

hyperParams.clipGradients = true;
hyperParams.maxGradNorm = 5;

%%% minFunc options: %%%

options.Method = 'lbfgs';
options.MaxFunEvals = 1000;
options.DerivativeCheck = 'on';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
% options.OutputFcn = @Display;  % Custom error reporting for minFunc

%%% TrainSGD learning options. %%%

options.numPasses = 1000;
options.miniBatchSize = 32;

% Learning parameters

% Choose AdaGrad or AdaDelta to compute parameter updates. 
% AdaDelta tends to find better solutions.
options.updateFn = @AdaDeltaUpdate;

% AdaDelta hyperparameters
options.adaDeltaRho = 0.95;
options.adaDeltaEps = 1e-7;

% AdaGrad hyperparameters
options.adaEps = 0.01;
options.lr = 0.05;

% How often (in steps) to report cost.
options.costFreq = 250;

% How often (in steps) to run on test data.
options.testFreq = 250;

% How often to report confusion matrices. 
% Should be a multiple of testFreq.
options.confusionFreq = 250;

% How often to display which items are misclassified.
% Should be a multiple of testFreq.
options.examplesFreq = 1000; 

% How often (in steps) to save parameters to disk.
% Checkpoints saved at these intervals coexist with checkpoints created
% automatically whenever a the best accuracy score is updated.
options.checkpointFreq = 100000; 

% The name assigned to the current call to TrainSGD. This can be used to
% distinguish multiple phases of training in the same experiment.
options.runName = 'tr';

% Reset the sum of squared gradients after this many iterations.
options.resetSumSqFreq = 10000;

end
