[data, wordMap, relationMap] = LoadConstitTrainingData();

V = size(wordMap, 1); % Num of words in vocab.
[theta, thetaDecoder, hyperParams] = InitializeModel(V);

data = InitializeWordFeatures(data, theta, thetaDecoder);

addpath('../minFunc/minFunc/')
addpath('../minFunc/minFunc/compiled/')
addpath('../minFunc/minFunc/mex/')
addpath('../minFunc/autoDif/')

% minfunc options (not tuned)
options.Method = 'lbfgs';
options.MaxIter = 200;
options.TolX = 1e-4;
options.DerivativeCheck = 'off';
options.Display = 'excessive';
options.numDiff = 0;
options.LS_init = '2' % Attempt to minimize evaluations per step...

% %%% PRETEST
% for i = 1:3
%     [classifierMatrices, classifierMatrix, classifierBias, ...
%         classifierParameters, wordFeatures] = stack2param(theta, thetaDecoder);
%     leftTree = data(i).leftTree;
%     rightTree = data(i).rightTree;
%     leftTree.updateFeatures(wordFeatures);
%     rightTree.updateFeatures(wordFeatures);
%     leftFeatures = leftTree.getFeatures();
%     rightFeatures = rightTree.getFeatures();
%     trueRelation = data(i).relation;
%     disp(leftTree.getText)
%     disp(rightTree.getText)
%     disp(trueRelation)
%     tensorInnerOutput = ComputeInnerTensorLayer(leftFeatures, rightFeatures, classifierMatrices, classifierMatrix, classifierBias);
%     tensorOutput = Sigmoid(tensorInnerOutput);
%     tensorDeriv = SigmoidDeriv(tensorInnerOutput);
%     relationProbs = ComputeSoftmaxProbabilities(tensorOutput, classifierParameters)
% end
% wordFeatures
% %%% END

theta = minFunc(@ComputeFullCostAndGrad, theta, options, thetaDecoder, data, hyperParams);

[cost, grad, acc, confusion] = ComputeFullCostAndGrad(theta, thetaDecoder, data, hyperParams);

acc
confusion

%  minFunc

%%% TEST
% for i = 1:3
%     [classifierMatrices, classifierMatrix, classifierBias, ...
%         classifierParameters, wordFeatures, compositionMatrices, ...
%     compositionMatrix, compositionBias] = stack2param(theta, thetaDecoder);
%     leftTree = data(i).leftTree;
%     rightTree = data(i).rightTree;
%     leftTree.updateFeatures(wordFeatures);
%     rightTree.updateFeatures(wordFeatures);
%     leftFeatures = leftTree.getFeatures();
%     rightFeatures = rightTree.getFeatures();
%     trueRelation = data(i).relation;
%     disp(leftTree.getText)
%     disp(rightTree.getText)
%     disp(trueRelation)
%     tensorInnerOutput = ComputeInnerTensorLayer(leftFeatures, rightFeatures, classifierMatrices, classifierMatrix, classifierBias);
%     tensorOutput = Sigmoid(tensorInnerOutput);
%     tensorDeriv = SigmoidDeriv(tensorInnerOutput);
%     relationProbs = ComputeSoftmaxProbabilities(tensorOutput, classifierParameters)
% end
wordFeatures(1:10,:)
