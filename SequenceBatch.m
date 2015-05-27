% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef SequenceBatch < handle
   % Note: This has been reasonably well optimize: Most of the complexity lies in
   % the layer funcitons themselves.

    properties
        B = -1;  % Batch size.
        D = -1;  % Number of dimensions in the feature vector at each position.
        N = -1;  % The length of the longest sequence that can be handled within this batch.
        sequences = [];  % The sequence objects in the batch.
        wordIndices = [];  % The index into the embedding matrix for each word in the sequence.
        wordCounts = [];  % The number of words in each sequence.
        features = [];  % All computed activation vectors.
                        % For speed, indexing is by (dim, batch-entry, sequence-index)
        rawEmbeddings = [];  % Word embeddings. Used only in conjunction with an embedding transform layer.
        inputFeatures = [];  % Same structure as features, but contains the
                                % the activations from the embedding transform layer if one is present.
        activationCache = [];  % Same structure as the bottom row of features, but contains the cached 
                               % IFOG activations for the LSTM.                                 
        masks = [];  % Same structure as transformInnerActivations, but contains dropout masks for the embedding transform layer.
        cFeatures = [];  % Same structure as features, but contains the cell activations.
        typeDummy = [];
    end
   
    methods(Static)
        function sb = makeSequenceBatch(sequences, wordFeatures, hyperParams)
            % Constructor: create and populate the batch data structures using a specific batch of data.
            % NOTE: This class is designed for use in a typical SGD setting (as in TrainSGD here) where batches are created, used once
            % and then destroyed. As such, this constructor bakes certain learned model parameters into the batch
            % object, and this any object created this way will become stale after one gradient step.

            sb = SequenceBatch();
            sb.B = length(sequences);
            sb.D = hyperParams.dim;

            % Find the length of the longest sequence. We use this to set the size of the main feature matrix,
            % to this value has a large impact on the run time of the batch.
            sb.N = max([sequences(:).wordCount]);

            sb.sequences = cell(sb.B, 1);  % TODO: Needed?

            sb.typeDummy = fZeros([1, 1], hyperParams.gpu);

            sb.wordIndices = fOnes([sb.N, sb.B], hyperParams.gpu && ~hyperParams.largeVocabMode);
            sb.wordCounts = [sequences(:).wordCount]';
            sb.inputFeatures = cell(sb.N, 1);
            sb.activationCache = cell(sb.N .* hyperParams.lstm, 1);
            sb.features = cell(sb.N, 1);
            sb.cFeatures = cell(sb.N .* hyperParams.lstm, 1);
            sb.masks = cell(sb.N .* hyperParams.useEmbeddingTransform, 1);
            sb.rawEmbeddings = cell(sb.N .* hyperParams.useEmbeddingTransform, 1);

            % Copy data in from the individual batch entries.
            for b = 1:sb.B                
                sb.sequences{b} = sequences(b);
                sb.wordIndices(sb.N - sb.wordCounts(b) + 1:end, b) = sequences(b).wordIndices;
            end

            for w = 1:sb.N
                % Populate the bottom row with word features.
                if hyperParams.useEmbeddingTransform > 0
                    if hyperParams.gpu && ~hyperParams.largeVocabMode
                        sb.rawEmbeddings{w} = gpuArray(wordFeatures(:, sb.wordIndices(w, :)));
                    else                     
                        sb.rawEmbeddings{w} = wordFeatures(:, sb.wordIndices(w, :));
                    end 
                    sb.rawEmbeddings{w} = bsxfun(@times, sb.rawEmbeddings{w}, [w >= sb.N - sb.wordCounts + 1]');
                else
                    sb.inputFeatures{w} = wordFeatures(:, sb.wordIndices(w, :));
                    sb.inputFeatures{w} = bsxfun(@times, sb.inputFeatures{w}, [w >= sb.N - sb.wordCounts + 1]');
                end
            end
        end
    end

    methods
        function [ endFeatures, connectionCosts, connectionAcc ] = runForward(sb, embeddingTransformMatrix, ~, ~, compositionMatrix, hyperParams, trainingMode)
            % Run the optional embedding transformation layer forward.
            % Recomputes features using fresh parameters.

            LSTM = size(compositionMatrix, 1) > size(compositionMatrix, 2);
            SUM =  isempty(compositionMatrix);

            for w = 1:sb.N
                if ~isempty(embeddingTransformMatrix)
                    % TODO: Adapt this zero-bias idea to LatticeBatch.
                    transformInputs = [ [w >= sb.N - sb.wordCounts + 1]'; sb.rawEmbeddings{w}(:, :)];
                    [ sb.inputFeatures{w}(:, :), sb.masks{w}(:, :) ] = ...
                        Dropout(tanh(embeddingTransformMatrix * transformInputs), hyperParams.bottomDropout, trainingMode, hyperParams.gpu);
                end

                % Compute a feature vector for the predecessor node.
                if w > 1
                    predActivations = sb.features{w - 1}(:, :);
                    if LSTM
                        predC = sb.cFeatures{w - 1}(:, :);
                    end
                else
                    if LSTM
                        predC = zeros([size(compositionMatrix, 1) / 4, sb.B], 'like', sb.typeDummy);
                        predActivations = zeros([size(compositionMatrix, 1) / 4, sb.B], 'like', sb.typeDummy);
                    else
                        predActivations = zeros([size(compositionMatrix, 1), sb.B], 'like', sb.typeDummy);
                    end
                end

                % Update the hidden features.
                if LSTM
                    [ sb.features{w}(:, :), sb.cFeatures{w}(:, :), sb.activationCache{w}(:, :) ] = ...
                        ComputeLSTMLayer(compositionMatrix, predActivations, predC, sb.inputFeatures{w}(:, :));
                elseif SUM
                    sb.features{w}(:, :) = predActivations + sb.inputFeatures{w}(:, :);
                else  % RNN
                    sb.features{w}(:, :) = ComputeRNNLayer(predActivations, sb.inputFeatures{w}(:, :), ...
                        compositionMatrix, @tanh);
                end
            end

            connectionCosts = 0;
            connectionAcc = -1;
            endFeatures = sb.features{sb.N};
        end

        function [ wordGradients, connectionMatrixGradients, scoringVectorGradients, ...
                   compositionMatrixGradients, embeddingTransformMatrixGradients ] = ...
            getGradient(sb, deltaH, wordFeatures, embeddingTransformMatrix, ~, ~, compositionMatrix, hyperParams)
            % Run backwards.

            LSTM = size(compositionMatrix, 1) > size(compositionMatrix, 2);
            SUM = isempty(compositionMatrix);

            if isempty(embeddingTransformMatrix)
                NUMTRANS = 0;
            else
                NUMTRANS = size(embeddingTransformMatrix, 3);
            end

            connectionMatrixGradients = zeros(0, 0, 'like', deltaH); 
            scoringVectorGradients = zeros(0, 0, 'like', deltaH);
            compositionMatrixGradients = fZeros(size(compositionMatrix), hyperParams.gpu);           
            embeddingTransformMatrixGradients = fZeros(size(embeddingTransformMatrix), hyperParams.gpu);
            if hyperParams.gpu && ~hyperParams.largeVocabMode
                wordGradients = fZeros(size(wordFeatures), hyperParams.gpu);
            else
                wordGradients = sparse([], [], [], ...
                    size(wordFeatures, 1), size(wordFeatures, 2), sb.N * sb.B);
            end

            if LSTM
                deltaC = 0 .* deltaH;
            end

            for w = sb.N:-1:1
                if w > 1
                    predActivations = sb.features{w - 1}(:, :);

                    if LSTM
                        predC = sb.cFeatures{w - 1}(:, :);
                    end
                else
                    if LSTM
                        predC = zeros([size(compositionMatrix, 1) / 4, sb.B], 'like', sb.typeDummy);
                        predActivations = zeros([size(compositionMatrix, 1) / 4, sb.B], 'like', sb.typeDummy);
                    else
                        predActivations = zeros([size(compositionMatrix, 1), sb.B], 'like', sb.typeDummy);
                    end
                end

                if LSTM
                    if w > 1
                        [ localCompositionMatrixGradients, compDeltaInput, deltaH, deltaC ] ...
                            = ComputeLSTMLayerGradients(sb.inputFeatures{w}(:, :), compositionMatrix, sb.activationCache{w}(:, :), ...
                                predC, predActivations, sb.cFeatures{w}(:, :), deltaH, deltaC);
                    else
                        [ localCompositionMatrixGradients, compDeltaInput ] ...
                            = ComputeLSTMLayerGradients(sb.inputFeatures{w}(:, :), compositionMatrix, sb.activationCache{w}(:, :), ...
                                predC, predActivations, sb.cFeatures{w}(:, :), deltaH, deltaC);      
                    end
                elseif SUM
                    compDeltaInput = deltaH;
                    localCompositionMatrixGradients = zeros(size(compositionMatrix), 'like', compositionMatrix);
                else  % RNN
                    [ localCompositionMatrixGradients, deltaH, compDeltaInput ] = ...
                    ComputeRNNLayerGradients(predActivations, sb.inputFeatures{w}(:, :), ...
                          compositionMatrix, deltaH, @TanhDeriv, sb.features{w}(:, :));
                end              
                
                if hyperParams.trainWords && NUMTRANS > 0
                    % Compute gradients for embedding transform layers and words
                    compDeltaInput = compDeltaInput .* sb.masks{w}(:, :); % Take dropout into account
                    compDeltaInput = bsxfun(@times, compDeltaInput, [w >= sb.N - sb.wordCounts + 1]');
                    [ localEmbeddingTransformMatrixGradients, compDeltaInput ] = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                              compDeltaInput, sb.rawEmbeddings{w}(:, :), ...
                              sb.inputFeatures{w}(:, :), @TanhDeriv, hyperParams.gpu);
                elseif NUMTRANS > 0
                    % Compute gradients for embedding transform layers only
                    compDeltaInput = compDeltaInput .* sb.masks{w}(:, :); % Take dropout into account
                    compDeltaInput = bsxfun(@times, compDeltaInput, [w >= sb.N - sb.wordCounts + 1]');
                    localEmbeddingTransformMatrixGradients = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                                compDeltaInput, sb.rawEmbeddings{w}(:, :), ...
                                sb.inputFeatures{w}(:, :), @TanhDeriv, hyperParams.gpu);
                end

                compositionMatrixGradients = ...
                    compositionMatrixGradients + ...
                    localCompositionMatrixGradients;

                if NUMTRANS > 0
                    embeddingTransformMatrixGradients = ...
                        embeddingTransformMatrixGradients + ...
                        localEmbeddingTransformMatrixGradients;    
                end

                % Push input deltas into the word gradients.
                if hyperParams.trainWords
                    if hyperParams.gpu && hyperParams.largeVocabMode
                        gathered = gather(compDeltaInput);
                        gathered = double(gathered);
                    else
                        gathered = compDeltaInput;
                    end
                    
                    % wordGradients = wordGradients + ...
                    %     CollectEmbeddingGradients(gathered, sb.wordIndices(w, :), size(wordGradients, 2));

                    for b = 1:sb.B
                        if w >= sb.N - sb.wordCounts(b) + 1
                            wordGradients(:, sb.wordIndices(w, b)) = ...
                                wordGradients(:, sb.wordIndices(w, b)) + ...
                                gathered(:, b);
                        end
                    end
                end 
            end
        end
    end
end
