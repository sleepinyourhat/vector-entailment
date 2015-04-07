% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef SequenceBatch < handle
   
    properties
        B = -1;  % Batch size.
        D = -1;  % Number of dimensions in the feature vector at each position.
        N = -1;  % The length of the longest sequence that can be handled within this pyramid.
        sequences = [];  % The sequence objects in the batch.
        wordIndices = [];  % The index into the embedding matrix for each word in the sequence.
        wordCounts = [];  % The number of words in each sequence.
        features = [];  % All computed activation vectors, with the words at the bottom row.
                        % For speed, indexing is by (dim, batch-entry, column, row)
        rawEmbeddings = [];  % Word embeddings. Used only in conjunction with an embedding transform layer.
        inputActivations = [];  % Same structure as the bottom row of features, but contains the
                                         % the pre-nonlinearity activations from the embedding transform layer
                                         % if one is present.
        masks = [];  % Same structure as transformInnerActivations, but contains dropout masks for the embedding transform layer.
        cFeatures = [];  % Same structure as features, but contains the cell activations.
    end
   
    methods(Static)
        function sb = makeSequenceBatch(sequneces, wordFeatures, hyperParams)
            % Constructor: create and populate the batch data structures using a specific batch of data.

            sb = SequenceBatch();
            sb.B = length(sequences);
            sb.D = hyperParams.dim;

            % Find the length of the longest sequence. We use this to set the size of the main feature matrix,
            % to this value has a large impact on the run time of the batch.
            sb.N = max([sequences(:).wordCount]);

            sb.sequences = cell(sb.B, 1);

            sb.wordIndices = zeros(sb.N, sb.B);
            sb.wordCounts = [pyramids(:).wordCount];
            sb.inputFeatures = zeros(sb.D, sb.B, sb.N);
            sb.features = zeros(sb.D, sb.B, sb.N);
            sb.cFeatures = zeros(sb.D, sb.B, sb.N);
            sb.rawEmbeddings = zeros(hyperParams.embeddingDim, sb.B, hyperParams.embeddingTransformDepth * sb.N);
            sb.masks = zeros(sb.D, sb.B, hyperParams.embeddingTransformDepth * sb.N);

            % Copy data in from the individual batch entries.
            for b = 1:sb.B
                sb.sequneces{b} = sequences(b);
                sb.wordIndices(sb.N - sb.wordCounts(b) + 1:end, b) = sequences(b).wordIndices;
                for w = sb.N - sb.wordCounts(b) + 1:end
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
        function [ topFeatures, connectionCosts ] = runForward(sb, embeddingTransformMatrix, connectionMatrix, compositionMatrix, hyperParams, trainingMode)
            % Run the optional embedding transformation layer forward.
            % Recomputes features using fresh parameters.

            LSTM = size(compositionMatrix, 1) > size(compositionMatrix, 2);

            for w = 1:pb.N
                % Compute a feature vector for the input at this node.
                if length(embeddingTransformMatrix) > 0
                    % Run the transform layer.
                    transformInnerActivations(:, :, col) = ...
                        embeddingTransformMatrix * transformInputs;
                    [ sb.features(:, :, col, pb.N), pb.masks(:, :, col) ] = ...
                        Dropout(tanh(pb.transformInnerActivations(:, :, col)), hyperParams.bottomDropout, trainingMode);

                    [obj.inputActivations, obj.mask] = Dropout(transformActivations, hyperParams.bottomDropout, trainingMode);
                end

                % Compute a feature vector for the predecessor node.
                if ~isempty(obj.pred)
                    obj.pred.updateFeatures(...
                        wordFeatures, compMatrices, compMatrix, embeddingTransformMatrix, ...
                        compNL, trainingMode, hyperParams);
                    
                    predActivations = obj.pred.activations;

                    if LSTM
                        predC = obj.pred.cActivations;
                    end
                else
                    if LSTM
                        predC = zeros(size(compMatrix, 1) / 4, 1);
                        predActivations = zeros(size(compMatrix, 1) / 4, 1);
                    else
                        predActivations = zeros(size(compMatrix, 1), 1);
                    end
                end    

                % Update the hidden features.
                if LSTM
                    [ obj.activations, obj.cActivations, obj.activationCache ] = ...
                        ComputeLSTMLayer(compMatrix, predActivations, predC, obj.inputActivations);
                else
                    obj.activations = ComputeRNNLayer(predActivations, obj.inputActivations,...
                        compMatrix, compNL);
                end
            end
        end

        function [ connectionClassifierInputs ] = collectConnectionClassifierInputs(sb, hyperParams, col, row)
            contextPositions = -hyperParams.pyramidConnectionContextWidth + 1:hyperParams.pyramidConnectionContextWidth;
            connectionClassifierInputs = zeros((2 * hyperParams.pyramidConnectionContextWidth) * sb.D + sb.NUMACTIONS, sb.B);

            for pos = 1:2 * hyperParams.pyramidConnectionContextWidth
                sourcePos = contextPositions(pos) + col;
                if (sourcePos > 0) && (sourcePos <= row + 1)
                    connectionClassifierInputs((pos - 1) * sb.D + 1:pos * sb.D, :) = sb.features(:, :, sourcePos, row + 1);
                else
                    connectionClassifierInputs((pos - 1) * sb.D + 1:pos * sb.D, :) = 0;
                end
                % Else: Leave in the zeros. Maybe fix replace with edge-of-sentence token? (TODO)
            end
            if col > 1
                connectionClassifierInputs(end - sb.NUMACTIONS + 1:end, :) = ...
                    sb.connections(:, :, col - 1, row);
            else
                connectionClassifierInputs(end - sb.NUMACTIONS + 1:end, :) = 0;
            end
        end

        function [ wordGradients, connectionMatrixGradients, ...
                   compositionMatrixGradients, embeddingTransformMatrixGradients ] = ...
            getGradient(sb, incomingDeltas, wordFeatures, embeddingTransformMatrix, connectionMatrix, compositionMatrix, hyperParams)
            % Run backwards.

            contextPositions = -hyperParams.pyramidConnectionContextWidth + 1:hyperParams.pyramidConnectionContextWidth;
            connectionMatrixGradients = zeros(size(connectionMatrix, 1), size(connectionMatrix, 2));
            compositionMatrixGradients = zeros(size(compositionMatrix, 1), size(compositionMatrix, 2));
            % Initialize delta matrix with the incoming deltas in the right places.

            % TODO: This could be represented as a vector covering only the deltas at one row,
            % but this could impose some time/complexity costs. Investigate.
            deltas = zeros(sb.D, sb.B, sb.N, sb.N);

            % Populate the delta matrix with the incoming deltas (reasonably fast).
            for b = 1:sb.B
                deltas(:, b, 1, sb.N - sb.wordCounts(b) + 1) = incomingDeltas(:, b);
            end

            % Initialize some variables that will be used inside the loop.
            deltasToConnections = zeros(sb.NUMACTIONS, sb.B);

            % Iterate over the structure in reverse
            for row = 1:sb.N - 1
                for col = row:-1:1
                    %% Handle composition function gradients %%

                    % Multiply in the three connection weights by the three inputs to the current features, 
                    % and add these to the existing deltas, which can either come from the inbound deltas (from the top-level classifier)
                    % or from a wide-window connection classifier.
                    deltas(:, :, col, row + 1) = ...
                        deltas(:, :, col, row + 1) + ...
                        bsxfun(@times, deltas(:, :, col, row), ...
                               sb.connections(1, :, col, row));
                    deltas(:, :, col + 1, row + 1) = ...
                        deltas(:, :, col + 1, row + 1) + ...
                        bsxfun(@times, deltas(:, :, col, row), ...
                               sb.connections(2, :, col, row));
                    compositionDeltas = ...
                        bsxfun(@times, deltas(:, :, col, row), ...
                               sb.connections(3, :, col, row));                 

                    % Backprop through the composition function.
                    [ localCompositionMatrixGradients, compositionDeltaLeft, compositionDeltaRight ] = ...
                        ComputeRNNLayerGradients(sb.features(:, :, col, row + 1), ...
                                                 sb.features(:, :, col + 1, row + 1), ...
                                                 compositionMatrix, compositionDeltas, @TanhDeriv, ...
                                                 sb.compositionInnerActivations(:, :, col, row));

                    % Add the composition gradients and deltas into the accumulators.
                    compositionMatrixGradients = compositionMatrixGradients + localCompositionMatrixGradients;
                    deltas(:, :, col, row + 1) = ...
                        deltas(:, :, col, row + 1) + compositionDeltaLeft;
                    deltas(:, :, col + 1, row + 1) = ...
                        deltas(:, :, col + 1, row + 1) + compositionDeltaRight;

                    %% Handle connection function gradients %%

                    % Multiply the deltas by the three inputs to the current features to compute deltas for the connections.
                    deltasToConnections(1, :) = deltasToConnections(1, :) + ...
                                                sum(deltas(:, :, col, row) .* ...
                                                 sb.features(:, :, col, row + 1), 1);
                    deltasToConnections(2, :) = deltasToConnections(2, :) + ...
                                                 sum(deltas(:, :, col, row) .* ...
                                                 sb.features(:, :, col + 1, row + 1), 1);
                    deltasToConnections(3, :) = deltasToConnections(3, :) + ...
                                                 sum(deltas(:, :, col, row) .* ...
                                                 sb.compositionActivations(:, :, col, row), 1);
                    
                    % Compute gradients from the connection classifier wrt. the incoming deltas from above and to the right.
                    [ localConnectionMatrixGradients, connectionDeltas ] = ...
                        ComputeBareSoftmaxGradients(connectionMatrix, sb.connections(:, :, col, row), ...
                            deltasToConnections, sb.connectionClassifierInputs(:, :, col, row));
                    connectionMatrixGradients = connectionMatrixGradients + localConnectionMatrixGradients;

                    % Compute gradients from the connection classifier wrt. the independent connection supervision signal.
                    [ localConnectionMatrixGradients, localConnectionDeltas ] = ...
                        ComputeSoftmaxClassificationGradients(connectionMatrix, sb.connections(:, :, col, row), ...
                            sb.connectionLabels(:, col, row), sb.connectionClassifierInputs(:, :, col, row), hyperParams, ...
                            sb.numNodes ./ hyperParams.connectionCostScale);
                    connectionMatrixGradients = connectionMatrixGradients + localConnectionMatrixGradients;

                    % Rescale by the number of times supervision is being applied.
                    connectionDeltas = connectionDeltas + localConnectionDeltas;

                    % Distribute the deltas from the softmax function back into its inputs.
                    for pos = 1:2 * hyperParams.pyramidConnectionContextWidth
                        sourcePos = contextPositions(pos) + col;
                        if sourcePos > 0 && sourcePos <= row + 1
                            deltas(:, :, sourcePos, row + 1) = ...
                                deltas(:, :, sourcePos, row + 1) + ...
                                connectionDeltas((pos - 1) * sb.D + 1:pos * sb.D, :);

                            multpliers = min(bsxfun(@rdivide, hyperParams.maxDeltaNorm, sum(deltas(:, :, sourcePos, row + 1).^2)), ones(1, sb.B));
                            deltas(:, :, sourcePos, row + 1) = bsxfun(@times, deltas(:, :, sourcePos, row + 1), multpliers);

                        end
                        if col > 1
                            deltasToConnections = bsxfun(@times, connectionDeltas(end + 1 - sb.NUMACTIONS:end, :), ...
                                                         permute(sb.activeNode(:, col, row), [2, 1, 3]));
                        else
                            deltasToConnections = zeros(sb.NUMACTIONS, sb.B);
                        end
                            
                    end
                end

                % Delete deltas for inactive nodes.
                deltas(:, :, :, row + 1) = ...
                       bsxfun(@times, deltas(:, :, :, row + 1) , ...
                           permute(sb.activeNode(:, :, row + 1), [3, 1, 2]));
            end

            % Run the embedding transform layers backwards.
            if hyperParams.embeddingTransformDepth > 0
                embeddingTransformMatrixGradients = zeros(size(embeddingTransformMatrix, 1), size(embeddingTransformMatrix, 2));                    rawEmbeddingDeltas = zeros(hyperParams.embeddingDim, sb.B, sb.N);
                rawEmbeddingDeltas = zeros(hyperParams.embeddingDim, sb.B, sb.N);
                for col = 1:sb.N
                    transformDeltas = deltas(:, :, col, sb.N) .* sb.masks(:, :, col); % Take dropout into account
                    [ localEmbeddingTransformMatrixGradients, rawEmbeddingDeltas(:, :, col) ] = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                              transformDeltas, sb.rawEmbeddings(:, :, col), ...
                              sb.transformInnerActivations(:, :, col), @TanhDeriv);
                    embeddingTransformMatrixGradients = embeddingTransformMatrixGradients + localEmbeddingTransformMatrixGradients;
                end
            else
                embeddingTransformMatrixGradients = [];
            end

            % Push deltas from the bottom row into the word gradients.
            % TODO: We don't need to require wordFeatures as an input, only used here.
            wordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), sb.N * sb.B);
            if hyperParams.trainWords
                for b = 1:sb.B
                    for col = 1:sb.wordCounts(b)
                        if hyperParams.embeddingTransformDepth > 0
                            wordGradients(:, sb.wordIndices(col, b)) = ...
                                wordGradients(:, sb.wordIndices(col, b)) + ...
                                rawEmbeddingDeltas(:, b, col);
                        else
                            wordGradients(:, sb.wordIndices(col, b)) = ...
                                wordGradients(:, sb.wordIndices(col, b)) + ...
                                deltas(:, b, col, sb.N);
                        end
                    end
                end
            end
        end
    end
end
