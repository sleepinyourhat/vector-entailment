function [ hyperParams, options ] = Defaults()
% Set defaults for all model and learning parameters.

% The dimensionality of the word/phrase vectors.
hyperParams.dim = 25;

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
hyperParams.lambda = 0.0002;

% L1 v. L2 regularization. If no regularization is needed, set
% lambda to 0 and ignore this parameter.
hyperParams.norm = 2;

% Use the syntactically untied composition layer params.
hyperParams.untied = false; 

% Use only the specified fraction of the training datasets
hyperParams.datasetsPortion = 1.0;

% Use only the specified fraction of the individual training examples
hyperParams.dataPortion = 1.0;

% When a dataset is to be automatically split into train and test portions,
% use this much as test data.
hyperParams.testFraction = 0.1;

% Use NTN layers in place of NN layers.
hyperParams.useThirdOrder = true;
hyperParams.useThirdOrderComparison = true;

% If set, train using minFunc. Only partially supported. See GradCheck for an example.
hyperParams.minFunc = false;

% Nonlinearities.
hyperParams.compNL = @Sigmoid;
hyperParams.compNLDeriv = @SigmoidDeriv; 
hyperParams.classNL = @LReLU;
hyperParams.classNLDeriv = @LReLUDeriv;

% If set, don't try to keep the entire training data set in memory at once.
hyperParams.fragmentData = false;

% If set, store embedding matrix gradients as spare matrices, and only apply regularization
% to the parameters that are in use at each step. This does nothing if trainWords is false.
% Useful as long as the vocabulary size is fairly large. (Exact threshold unknown.)
hyperParams.fastEmbed = false;

%%% minFunc options: %%%

options.Method = 'lbfgs';
options.MaxFunEvals = 1000;
options.DerivativeCheck = 'on';
options.Display = 'full';
options.numDiff = 0;
options.LS_init = '2'; % Attempt to minimize evaluations per step...
options.PlotFcns = [];
% options.OutputFcn = @Display;  % Custom error reporting for minFunc

%%% AdaGradSGD learning options. %%%

options.numPasses = 250;
options.miniBatchSize = 64;

% AdaGrad LR
options.lr = 0.05;

% How often (in steps) to report cost.
options.costFreq = 200;

% How often (in steps) to run on test data.
options.testFreq = 100;

% How often to report confusion matrices. 
% Should be a multiple of testFreq.
options.confusionFreq = 100;

% How often to display which items are misclassified.
% Should be a multiple of testFreq.
options.examplesFreq = 800; 

% How often (in steps) to save parameters to disk.
options.checkpointFreq = 2000; 

% The name assigned to the current call to AdaGradSGD. This can be used to
% distinguish multiple phases of training in the same experiment.
options.runName = 'tr';

% Reset the sum of squared gradients after this many iterations.
options.resetSumSqFreq = 3200;


end