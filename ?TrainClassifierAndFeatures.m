function TrainClassifierAndFeatures(data)
disp('Starting training.')

global DIM;
DIM_COMBINED = 2 * DIM;
global NUM_RELATIONS;
NUM_RELATIONS = 7;

global PENULT_DIM;
PENULT_DIM = 10;

global LAMBDA;
LAMBDA = .001;

% Learning rate params.
ALPHA_CEIL = 1;
ALPHA_DECR = 0.9;

% SGD params.
MINIBATCH_SIZE = 10;
ERROR_INTERVAL = length(data);

% How much can a gradient be off before we complain?
global TEST_EPLISON
TEST_EPSILON = 0.01;

global wordFeatures; % Columns=words; Rows=dimensions
global classifierParameters;
global classifierMatrices;
global classifierMatrix;
global classifierBias;

notConverged = 1;
cycleCount = 1;

errors = zeros(1, 1000);
countErrors = zeros(1, 1000);
alpha = ALPHA_CEIL;

% Initialize the gradients.
softmaxGradient = zeros(NUM_RELATIONS, PENULT_DIM + 1);
localSoftmaxGradient = zeros(NUM_RELATIONS, PENULT_DIM + 1);
clMatricesGradients = zeros(DIM , DIM * PENULT_DIM);
localClMatricesGradients = zeros(DIM , DIM * PENULT_DIM);
clMatrixGradients = zeros(PENULT_DIM, 2 * DIM );
localClMatrixGradients = zeros(PENULT_DIM, 2 * DIM);
clBiasGradients = zeros(PENULT_DIM, 1);
localClBiasGradients = zeros(PENULT_DIM, 1);
wordGradients = sparse([], [], [], size(wordFeatures, 1), size(wordFeatures, 2), length(data));
localWordGradients = sparse([], [], [], size(wordFeatures, 1), size(wordFeatures, 2), 10);

% Initialize the error trackers.
error = 0;
countError = 0;

% Initialize the counter.
dataItemsSeen = 0;

while notConverged
    % Whenever we start a pass through the data, shuffle the order.
    if mod(dataItemsSeen, length(data)) == 0;
        dataOrder = randperm(length(data));
        orderInd = 1;
        
        % And decrement alpha.
        alpha = alpha * ALPHA_DECR;
    end
    
    % TODO: Parallelize this.
    
    % Choose a data item.
    dataInd = dataOrder(orderInd);
    dataItemsSeen = dataItemsSeen + 1;
    
    leftTree = data(dataInd).leftTree;
    rightTree = data(dataInd).rightTree;
    trueRelation = data(dataInd).relation;
    
    leftFeatures = leftTree.getFeatures();
    rightFeatures = rightTree.getFeatures();
    
    % Use the tensor layer to build classifier input:
    tensorInnerOutput = ComputeInnerTensorLayer(leftFeatures, rightFeatures, classifierMatrices, classifierMatrix, classifierBias);
    tensorOutput = Sigmoid(tensorInnerOutput);
    tensorDeriv = SigmoidDeriv(tensorInnerOutput);
    
    relationProbs = ComputeSoftmaxProbabilities(tensorOutput, classifierParameters);
    
    % Increment local error
    error = error + Objective(trueRelation, relationProbs);
    [~, prediction] = max(relationProbs);
    countError = countError + (prediction ~= trueRelation);
    
    % Compute node softmax error, mid-left of p6 of tensor paper
    targetRelationProbs = zeros(length(relationProbs), 1);
    targetRelationProbs(trueRelation) = 1;
    softmaxDeltaFirstHalf = classifierParameters' * (relationProbs - targetRelationProbs);
    softmaxDeltaSecondHalf = SigmoidDeriv([1; tensorOutput]); % Intercept
    softmaxDelta = (softmaxDeltaFirstHalf .* softmaxDeltaSecondHalf);
    
    parfor relEval = 1:NUM_RELATIONS
        % Del from ufldl wiki on softmax
        localSoftmaxGradient(relEval, :) = ...
            -([1; tensorOutput] .* ...
            ((trueRelation == relEval) - relationProbs(relEval)))';
    end
    
    % Test gradients
    testi = randi(size(classifierParameters, 1));
    testj = randi(size(classifierParameters, 2));
    tempLocalSoftmaxGradient = localSoftmaxGradient + LAMBDA * classifierParameters;
    softmaxGradientError = ...
        TestGradientInSoftmax(testi, testj, ...
        tempLocalSoftmaxGradient(testi,testj), classifierParameters, ...
        tensorOutput, trueRelation);
    if softmaxGradientError > 1 + TEST_EPSILON || softmaxGradientError < 1 - TEST_EPSILON
        disp('Softmax gradient inaccurate.')
    end
    
    softmaxGradient = softmaxGradient + localSoftmaxGradient;
    
    % Calculate third order tensor gradients for tensor layer
    for (relEval = 1:PENULT_DIM)
        Cols = (DIM*(relEval - 1))+1:(DIM*relEval);
        localClMatricesGradients(:,Cols) = (tensorDeriv(relEval) * softmaxDelta(relEval + 1)) .* (leftFeatures' * rightFeatures);
    end
    
    % Calculate matrix gradients for tensor layer
    parfor (relEval = 1:PENULT_DIM)
        localClMatrixGradients(relEval, :) = (tensorDeriv(relEval) * softmaxDelta(relEval + 1)) .* [leftFeatures rightFeatures]';
    end
    
    % Calculate vector gradients for tensor layer
    localClBiasGradients = (tensorDeriv .* softmaxDelta(2:PENULT_DIM + 1));
    
    % Test gradients
    [params, decoder] = param2stack(classifierMatrices, classifierMatrix, classifierBias, classifierParameters);
    
    testi = randi(prod(size(classifierMatrices)));
    tensorGradientError = TestGradientInTensor(testi, ...
        localClMatricesGradients(testi) + (LAMBDA * params(testi)), params, decoder, leftFeatures, ...
        rightFeatures, trueRelation);
    if tensorGradientError > 1 + TEST_EPSILON || tensorGradientError < 1 - TEST_EPSILON
        disp('Tensor layer tensor gradient inaccurate.')
        tensorGradientError
    end
    
    localtesti = randi(prod(size(classifierMatrix)));
    paramstesti = prod(size(classifierMatrices)) + localtesti;
    % Expect that params(paramstesti) == classifierMatrix(localtesti)
    clMatrixGradientError = TestGradientInTensor(paramstesti, ...
        localClMatrixGradients(localtesti) + (LAMBDA * params(paramstesti)) , params, decoder, leftFeatures, ...
        rightFeatures, trueRelation);
    if clMatrixGradientError > 1 + TEST_EPSILON || clMatrixGradientError < 1 - TEST_EPSILON
        disp('Tensor layer matrix gradient inaccurate.')
        clMatrixGradientError
    end
    
    localtesti = randi(prod(size(classifierBias)));
    paramstesti = prod(size(classifierMatrices)) + prod(size(classifierMatrix)) + localtesti;
    % Expect that params(paramstesti) == classifierMatrix(localtesti)
    clMatrixGradientError = TestGradientInTensor(paramstesti, ...
        localClBiasGradients(localtesti) + (LAMBDA * params(paramstesti)) , params, decoder, leftFeatures, ...
        rightFeatures, trueRelation);
    if clMatrixGradientError > 1 + TEST_EPSILON || clMatrixGradientError < 1 - TEST_EPSILON
        disp('Tensor layer bias gradient inaccurate.')
        clMatrixGradientError
    end
    
    clMatricesGradients = clMatricesGradients + localClMatricesGradients;
    
    clMatrixGradients = clMatrixGradients + localClMatrixGradients;
    
    % Compute word gradients
    
    % Choose words to compute gradients for
    % In the two-word example case, this is trivial
    
    % Left word ONLY here
    %         localWordGradients(leftTree.getWordIndex, :) = zeros(1, DIM) + 1;
    %
    %         [words, worddecoder] = param2stack(leftFeatures, rightFeatures);
    %         testi = randi(DIM);
    %         wordGradientError = TestGradientInWord(testi, ...
    %             localWordGradients(testi) + (LAMBDA * words(testi)) , params, ...
    %             decoder, words, worddecoder, trueRelation)
    %         if wordGradientError > 1.01 || wordGradientError < 0.99
    %             disp('Word gradient inaccurate.')
    %         end
    
    % Start of right col
    %
    %             S = zeros(DIM_COMBINED, 1);
    %             for (relEval = 1:NUM_RELATIONS)
    %                 Cols = (DIM_COMBINED*(relEval-1)) + 1:(DIM_COMBINED*relEval);
    %                 firstHalf = softmaxDelta(relEval) .* (sliceGradients(:,Cols) + sliceGradients(:,Cols)');
    %                 secondHalf = combinedFeatures;
    %                 S = S + (firstHalf * secondHalf');
    %             end
    %
    %             messageDown = S .* SigmoidDeriv(combinedFeatures)';
    %
    %             % In this setup, messageDown goes straight to the words.
    %
    %             testi = randi(size(localSliceGradients, 1));
    %             wordFeatureGradientError = TestGradientInCombinedFeatures(testi, ...
    %                 messageDown(testi), classifierMatrices, ...
    %                 classifierParameters, combinedFeatures, trueRelation);
    %             if wordFeatureGradientError > 10 || wordFeatureGradientError < -1
    %             	disp('Word feature layer gradient inaccurate.')
    %             end
    %
    %             leftVocabIndex = leftTree.getWordIndex();
    %             rightVocabIndex = rightTree.getWordIndex();
    %
    %             wordGradients(leftVocabIndex, :) = messageDown(1:DIM);
    %             wordGradients(rightVocabIndex, :) = messageDown(DIM + 1:2 * DIM);
    
    
    % When we finish a minibatch, update.
    if mod(dataItemsSeen, MINIBATCH_SIZE) == 0
        
        % Finish building the gradient and apply it:
        softmaxGradient = softmaxGradient ./ length(data);
        softmaxGradient = softmaxGradient + (LAMBDA .* classifierParameters);
        classifierParameters = classifierParameters - (ALPHA_CEIL .* softmaxGradient);
        
        % Do the same for the tensor layer.
        clMatricesGradients = clMatricesGradients ./ MINIBATCH_SIZE;
        clMatricesGradients = clMatricesGradients + (LAMBDA .* classifierMatrices);
        classifierMatrices = classifierMatrices - (ALPHA_CEIL .* clMatricesGradients);
        clMatrixGradients = clMatrixGradients ./ MINIBATCH_SIZE;
        clMatrixGradients = clMatrixGradients + (LAMBDA .* classifierMatrix);
        classifierMatrix = classifierMatrix - (ALPHA_CEIL .* clMatrixGradients);
        clBiasGradients = clBiasGradients ./ MINIBATCH_SIZE;
        clBiasGradients = clBiasGradients + (LAMBDA .* classifierBias);
        classifierBias = classifierBias - (ALPHA_CEIL .* clBiasGradients);
        
        % Do the same for the words.
        wordFeatures = wordFeatures - (ALPHA_CEIL .* wordGradients);
        
        % Update the trees.
        parfor dataInd = 1:length(data)
            data(dataInd).leftTree.updateFeatures();
            data(dataInd).leftTree.updateFeatures();
        end
        
        % Reset the gradients.
        softmaxGradient = zeros(NUM_RELATIONS, PENULT_DIM + 1);
        localSoftmaxGradient = zeros(NUM_RELATIONS, PENULT_DIM + 1);
        clMatricesGradients = zeros(DIM , DIM * PENULT_DIM);
        localClMatricesGradients = zeros(DIM , DIM * PENULT_DIM);
        clMatrixGradients = zeros(PENULT_DIM, 2 * DIM );
        localClMatrixGradients = zeros(PENULT_DIM, 2 * DIM);
        clBiasGradients = zeros(PENULT_DIM, 1);
        localClBiasGradients = zeros(PENULT_DIM, 1);
        wordGradients = sparse([], [], [], size(wordFeatures, 1), size(wordFeatures, 2), length(data));
        localWordGradients = sparse([], [], [], size(wordFeatures, 1), size(wordFeatures, 2), 10);
    end
    
    if mod(dataItemsSeen, ERROR_INTERVAL) == 0 
        
                % TODO: Make this into a real convergence test.
        
  
        % Update the error.
        params = param2stack(classifierMatrices, classifierBias, classifierMatrices, classifierParameters, wordFeatures);
        error = (error / ERROR_INTERVAL) + Objective(1, [1], params);
        countError = countError / ERROR_INTERVAL;
        disp('======');
        disp(['Seen ', num2str(dataItemsSeen), ' items']);
        disp(['Showing error from last ', num2str(ERROR_INTERVAL)]);
        disp(['Error: ', num2str(error)]);
        disp(['Percent training items incorrect: ', num2str(countError)]);
        
        % Reset the error trackers.
        error = 0;
        countError = 0;
    end
    
    orderInd = orderInd + 1;
            
        %TODO: Convergence test?
        if (dataItemsSeen >= 10000)
            notConverged = 0;
        end
end

end


