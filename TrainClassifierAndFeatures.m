function TrainClassifierAndFeatures(data)
    disp('Starting training.')

	global DIM;
    DIM_COMBINED = 2 * DIM;
	global NUM_RELATIONS;
    NUM_RELATIONS = 7;

    LAMBDA = 0;

	% Columns=words; Rows=dimensions
	global wordFeatures;

	global classifierParameters;

    global classifierMatrices;
    
	% Todo: Build up tree to two top nodes

	notConverged = 1;
	cycleCount = 1;
    
    % 
    % For histogram of error:
    % global gradient_errors
    % gradient_errors = [];
    
    errors = zeros(1, 1000);
    countErrors = zeros(1, 1000);
    
	while notConverged,

		% Run one cycle through the data.
		softmaxGradient = zeros(NUM_RELATIONS, NUM_RELATIONS);
		localSoftmaxGradient = zeros(NUM_RELATIONS, NUM_RELATIONS);

        sliceGradients = zeros((DIM * 2) , ((DIM * 2)) * NUM_RELATIONS);
        localSliceGradients = zeros((DIM * 2) , ((DIM * 2) ) * NUM_RELATIONS);

        wordGradients = zeros(size(wordFeatures, 1), size(wordFeatures, 2));
        
        error = 0;
        countError = 0;
        
		% TODO: Parallelize lots of this.
        % TODO: Switch from batch to SGD
		for dataInd = 1:length(data),
			leftTree = data(dataInd).leftTree;
			rightTree = data(dataInd).rightTree;
			trueRelation = data(dataInd).relation;

            leftFeatures = leftTree.getFeatures();
            rightFeatures = rightTree.getFeatures();
            
            % Use the tensor layer to build classifier input:
            combinedFeatures = [leftFeatures, rightFeatures];
            
            tensorOutput = ComputeTensorLayer(combinedFeatures, classifierMatrices);
            
            relationProbs = ComputeSoftmaxProbabilities(tensorOutput, classifierParameters);
            
            error = error + Objective(trueRelation, relationProbs);
    
            % Add to the count error
            [dummy prediction] = max(relationProbs);
            countError = countError + (prediction ~= trueRelation);
            
            % Compute node softmax error, mid-left of p6 of tensor paper
            targetRelationProbs = zeros(length(relationProbs), 1);
            targetRelationProbs(trueRelation) = 1;
            softmaxDeltaFirstHalf = classifierParameters' * (relationProbs - targetRelationProbs); 
            softmaxDeltaSecondHalf = SigmoidDeriv(tensorOutput);
            softmaxDelta = (softmaxDeltaFirstHalf .* softmaxDeltaSecondHalf);
            
            for (relEval = 1:NUM_RELATIONS)
                % Delta from ufldl wiki on softmax
                localSoftmaxGradient(relEval, :) = ...
                    -(tensorOutput .* ...
                       ((trueRelation == relEval) - relationProbs(relEval)))'; 
            end
            
            testi = randi(size(classifierParameters, 1));
            testj = randi(size(classifierParameters, 2));
            softmaxGradientError = ...
                TestGradientInSoftmax(testi, testj, ...
                localSoftmaxGradient(testi,testj), classifierParameters, ...
                tensorOutput, trueRelation);
            if softmaxGradientError > 1.75 || softmaxGradientError < 0.75
                disp('Softmax gradient inaccurate.')
            end
                   
            softmaxGradient = softmaxGradient + localSoftmaxGradient;
            
            % End of p6 left col
            for (relEval = 1:NUM_RELATIONS)
                Cols = (DIM_COMBINED*(relEval - 1))+1:(DIM_COMBINED*relEval);
                localSliceGradients(:,Cols) = softmaxDelta(relEval) * combinedFeatures' * combinedFeatures;
            end
            
            testi = randi(size(localSliceGradients, 1));
            testj = randi(size(localSliceGradients, 2));
            tensorGradientError = TestGradientInTensor(testi, testj, ...
                localSliceGradients(testi,testj), classifierMatrices, ...
                classifierParameters, combinedFeatures, trueRelation);
            if tensorGradientError > 100 || tensorGradientError < -1
            	disp('Tensor layer gradient inaccurate.')
            end
            
            sliceGradients = sliceGradients + localSliceGradients;

            % Start of right col
            
            S = zeros(DIM_COMBINED, 1);
            for (relEval = 1:NUM_RELATIONS)
                Cols = (DIM_COMBINED*(relEval-1)) + 1:(DIM_COMBINED*relEval);
                firstHalf = softmaxDelta(relEval) .* (sliceGradients(:,Cols) + sliceGradients(:,Cols)');
                secondHalf = combinedFeatures;
                S = S + (firstHalf * secondHalf');
            end
            
            messageDown = S .* SigmoidDeriv(combinedFeatures)';
            
            S 
            messageDown
            
            % In this setup, messageDown goes straight to the words.

            testi = randi(size(localSliceGradients, 1));
            wordFeatureGradientError = TestGradientInCombinedFeatures(testi, ...
                messageDown(testi), classifierMatrices, ...
                classifierParameters, combinedFeatures, trueRelation);
            if wordFeatureGradientError > 10 || wordFeatureGradientError < -1
            	disp('Word feature layer gradient inaccurate.')
            end
            
            leftVocabIndex = leftTree.getWordIndex();
            rightVocabIndex = rightTree.getWordIndex();
            
            wordGradients(leftVocabIndex, :) = messageDown(1:DIM);
            wordGradients(rightVocabIndex, :) = messageDown(DIM + 1:2 * DIM);
            
            if error == -Inf
                break
            end
		end
	
	    % Finish building the gradient:
	    softmaxGradient = softmaxGradient ./ length(data);
	    
	    % Add the weight decay term:
		% softmaxGradient = softmaxGradient + (LAMBDA .* classifierParameters);
	
        % Add the weight penalty
        %error = error + LAMBDA * norm(classifierParameters, 2)
        %              + norm(wordFeatures, 2);
        
        countError = countError / length(data)
        countErrors(cycleCount) = countError;
                      	
		% Apply the gradient.
        classifierParameters = classifierParameters - softmaxGradient;

        
        % Do the same for the tensor slices.
        sliceGradients = sliceGradients ./ length(data);
        classifierMatrices = classifierMatrices - sliceGradients;

        % Do the same for the words.
        wordFeatures = wordFeatures - wordGradients;

		% Update the trees.
		for dataInd = 1:length(data)
			data(dataInd).leftTree.updateFeatures();
			data(dataInd).leftTree.updateFeatures();
		end

		cycleCount = cycleCount + 1;
		
        error = error / length(data)
        errors(cycleCount) = error;
        
		% TODO: Make this into a real convergence test.
		disp(['Finished training cycle ', num2str(cycleCount)])
		if (cycleCount >= 1000)
		    notConverged = 0;
        end
    end
end
