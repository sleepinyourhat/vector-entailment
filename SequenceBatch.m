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

            sb.wordIndices = zeros(sb.N, sb.B);
            sb.wordCounts = [sequences(:).wordCount]';
            sb.inputFeatures = zeros(sb.D, sb.B, sb.N);
            sb.rawEmbeddings = zeros(hyperParams.embeddingDim, sb.B, hyperParams.embeddingTransformDepth * sb.N);
            sb.activationCache = zeros(sb.D * 4, sb.B, sb.N);
            sb.features = zeros(sb.D, sb.B, sb.N);
            sb.cFeatures = zeros(sb.D, sb.B, sb.N);
            sb.masks = zeros(sb.D, sb.B, hyperParams.embeddingTransformDepth * sb.N);

            % Copy data in from the individual batch entries.
            for b = 1:sb.B                
                sb.sequences{b} = sequences(b);
                sb.wordIndices(sb.N - sb.wordCounts(b) + 1:end, b) = sequences(b).wordIndices;
                for w = sb.N - sb.wordCounts(b) + 1:sb.N
                    % Populate the bottom row with word features.

                    if hyperParams.embeddingTransformDepth > 0
                        sb.rawEmbeddings(:, b, w) = wordFeatures(:, sb.wordIndices(w, b));
                    else
                        sb.inputFeatures(:, b, w) = wordFeatures(:, sb.wordIndices(w, b));
                    end
                end
            end
        end
    end

    methods
        function [ endFeatures, connectionCosts, connectionAcc ] = runForward(sb, embeddingTransformMatrix, ~, compositionMatrix, hyperParams, trainingMode)
            % Run the optional embedding transformation layer forward.
            % Recomputes features using fresh parameters.

            LSTM = size(compositionMatrix, 1) > size(compositionMatrix, 2);
            SUM =  isempty(compositionMatrix);

            for w = 1:sb.N
                if ~isempty(embeddingTransformMatrix)
                    % TODO: Adapt this zero-bias idea to LatticeBatch.
                    transformInputs = [ [w >= sb.N - sb.wordCounts + 1]'; sb.rawEmbeddings(:, :, w)];
                    [ sb.inputFeatures(:, :, w), sb.masks(:, :, w) ] = ...
                        Dropout(tanh(embeddingTransformMatrix * transformInputs), hyperParams.bottomDropout, trainingMode);
                end

                % Compute a feature vector for the predecessor node.
                if w > 1
                    predActivations = sb.features(:, :, w - 1);
                    if LSTM
                        predC = sb.cFeatures(:, :, w - 1);
                    end
                else
                    if LSTM
                        predC = zeros(size(compositionMatrix, 1) / 4, sb.B);
                        predActivations = zeros(size(compositionMatrix, 1) / 4, sb.B);
                    else
                        predActivations = zeros(size(compositionMatrix, 1), sb.B);
                    end
                end

                % Update the hidden features.
                if LSTM
                    [ sb.features(:, :, w), sb.cFeatures(:, :, w), sb.activationCache(:, :, w) ] = ...
                        ComputeLSTMLayer(compositionMatrix, predActivations, predC, sb.inputFeatures(:, :, w));
                elseif SUM
                    sb.features(:, :, w) = predActivations + sb.inputFeatures(:, :, w);
                else  % RNN
                    sb.features(:, :, w) = ComputeRNNLayer(predActivations, sb.inputFeatures(:, :, w), ...
                        compositionMatrix, @tanh);
                end
            end

            connectionCosts = 0;
            connectionAcc = -1;
            endFeatures = sb.features(:, :, sb.N);
        end

        function [ wordGradients, connectionMatrixGradients, ...
                   compositionMatrixGradients, embeddingTransformMatrixGradients ] = ...
            getGradient(sb, deltaH, wordFeatures, embeddingTransformMatrix, ~, compositionMatrix, hyperParams)
            % Run backwards.

            LSTM = size(compositionMatrix, 1) > size(compositionMatrix, 2);
            SUM = isempty(compositionMatrix);

            HIDDENDIM = size(deltaH, 1);
            EMBDIM = size(wordFeatures, 1);
            if isempty(embeddingTransformMatrix)
                NUMTRANS = 0;
            else
                NUMTRANS = size(embeddingTransformMatrix, 3);
            end

            connectionMatrixGradients = []; 
            compositionMatrixGradients = zeros(size(compositionMatrix, 1), size(compositionMatrix, 2), size(compositionMatrix, 3));           
            embeddingTransformMatrixGradients = zeros(HIDDENDIM, EMBDIM + 1, NUMTRANS);
            wordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), sb.N * sb.B);

            if LSTM
                deltaC = 0 .* deltaH;
            end

            for w = sb.N:-1:1
                if w > 1
                    predActivations = sb.features(:, :, w - 1);

                    if LSTM
                        predC = sb.cFeatures(:, :, w - 1);
                    end
                else
                    if LSTM
                        predC = zeros(size(compositionMatrix, 1) / 4, sb.B);
                        predActivations = zeros(size(compositionMatrix, 1) / 4, sb.B);
                    else
                        predActivations = zeros(size(compositionMatrix, 1), sb.B);
                    end
                end

                if LSTM
                    if w > 1
                        [ localCompositionMatrixGradients, compDeltaInput, deltaH, deltaC ] ...
                            = ComputeLSTMLayerGradients(sb.inputFeatures(:, :, w), compositionMatrix, sb.activationCache(:, :, w), ...
                                predC, predActivations, sb.cFeatures(:, :, w), deltaH, deltaC);
                    else
                        [ localCompositionMatrixGradients, compDeltaInput ] ...
                            = ComputeLSTMLayerGradients(sb.inputFeatures(:, :, w), compositionMatrix, sb.activationCache(:, :, w), ...
                                predC, predActivations, sb.cFeatures(:, :, w), deltaH, deltaC);      
                    end
                elseif SUM
                    compDeltaInput = deltaH;
                    localCompositionMatrixGradients = zeros(size(compositionMatrix, 1), size(compositionMatrix, 2), size(compositionMatrix, 3));
                else  % RNN
                    [ localCompositionMatrixGradients, deltaH, compDeltaInput ] = ...
                    ComputeRNNLayerGradients(predActivations, sb.inputFeatures(:, :, w), ...
                          compositionMatrix, deltaH, @TanhDeriv, sb.features(:, :, w));
                end              
                
                if hyperParams.trainWords && NUMTRANS > 0
                    % Compute gradients for embedding transform layers and words
                    compDeltaInput = compDeltaInput .* sb.masks(:, :, w); % Take dropout into account
                    compDeltaInput = bsxfun(@times, compDeltaInput, [w >= sb.N - sb.wordCounts + 1]');
                    [ localEmbeddingTransformMatrixGradients, compDeltaInput ] = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                              compDeltaInput, sb.rawEmbeddings(:, :, w), ...
                              sb.inputFeatures(:, :, w), @TanhDeriv);
                elseif NUMTRANS > 0
                    % Compute gradients for embedding transform layers only
                    compDeltaInput = compDeltaInput .* sb.masks(:, :, w); % Take dropout into account
                    compDeltaInput = bsxfun(@times, compDeltaInput, [w >= sb.N - sb.wordCounts + 1]');
                    localEmbeddingTransformMatrixGradients = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                                compDeltaInput, sb.rawEmbeddings(:, :, w), ...
                                sb.inputFeatures(:, :, w), @TanhDeriv);
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
                    for b = 1:sb.B
                        if w >= sb.N - sb.wordCounts(b) + 1
                            wordGradients(:, sb.wordIndices(w, b)) = ...
                                wordGradients(:, sb.wordIndices(w, b)) + ...
                                compDeltaInput(:, b);
                        end
                    end
                end 
            end
        end
    end
end
