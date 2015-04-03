% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef PyramidBatch < handle
   
    properties
        NUMACTIONS = 3;  % The number of connection types.
                         % 1 := Copy left child.
                         % 2 := Copy right child.
                         % 3 := Compose left and right children.
        B = -1;  % Batch size.
        D = -1;  % Number of dimensions in the feature vector at each position.
        N = -1;  % The length of the longest sequence that can be handled within this pyramid.
        R = -1;  % The number of rows in features and deltas.
        pyramids = [];  % The pyramid objects in the batch.
        wordIndices = [];  % The index into the embedding matrix for each word in the sequence.
        wordCounts = [];  % The number of words in each sequence.
        features = [];  % All computed activation vectors, with the words at the bottom row.
                        % For speed, indexing is by (dim, batch-entry, column, row)
        rawEmbeddings = [];  % Word embeddings. Used only in conjunction with an embedding transform layer.
        transformInnerActivations = [];  % Same structure as the bottom row of features, but contains the
                                         % the pre-nonlinearity activations from the embedding transform layer
                                         % if one is present.
        masks = [];  % Same structure as transformInnerActivations, but contains dropout masks for the embedding transform layer.
        compositionInnerActivations = [];  % Same structure as features, but contains the pre-nonlinearity
                                           % activations from the composition function, and has no bottom (word) layer.
        compositionActivations = [];  % Same structure as features, but contains the
                                      % activations from the composition function, and has no bottom (word) layer.
        connections = [];  % The length-3 vectors of weights for the three connection types at each position in the pyramid.
                           % Has no bottom (word) layer.
        connectionLabels = [];  % The optional correct connection type (in {1, 2, 3}) for each position in the pyramid.
        activeNode = [];  % Triangular boolean matrix for each batch entry indicating whether each position 
                          % is within the pyramid structure for that entry.
        numNodes = []; % The number of active nodes for each pyramid.

        % TODO: Clip deltas as they are created.
    end
   
    methods(Static)
        function pb = makePyramidBatch(pyramids, wordFeatures, hyperParams)
            % Constructor: create and populate the batch data structures using a specific batch of data.

            pb = PyramidBatch();
            pb.B = length(pyramids);
            pb.D = hyperParams.dim;

            % Find the length of the longest sequence. We use this to set the size of the main feature matrix,
            % to this value has a large impact on the run time of the batch.
            pb.N = max([pyramids(:).wordCount]);

            pb.pyramids = cell(pb.B, 1);

            pb.wordIndices = zeros(pb.N, pb.B);
            pb.wordCounts = [pyramids(:).wordCount];
            pb.features = zeros(pb.D, pb.B, pb.N, pb.N);
            pb.rawEmbeddings = zeros(hyperParams.embeddingDim, pb.B, hyperParams.embeddingTransformDepth * pb.N);
            pb.transformInnerActivations = zeros(pb.D, pb.B, hyperParams.embeddingTransformDepth * pb.N);
            pb.masks = zeros(pb.D, pb.B, hyperParams.embeddingTransformDepth * pb.N);
            pb.compositionInnerActivations = zeros(pb.D, pb.B, pb.N - 1, pb.N - 1);
            pb.compositionActivations = zeros(pb.D, pb.B, pb.N - 1, pb.N - 1);
            pb.connections = zeros(pb.NUMACTIONS, pb.B, pb.N - 1, pb.N - 1);
            pb.connectionLabels = zeros(pb.B, pb.N - 1, pb.N - 1);
            pb.activeNode = zeros(pb.B, pb.N, pb.N);
            pb.numNodes = (pb.wordCounts' - 1) .^ 2;

            % Copy data in from the individual batch entries.
            for b = 1:pb.B
                pb.pyramids{b} = pyramids(b);
                pb.wordIndices(1:pb.wordCounts(b), b) = pyramids(b).wordIndices;
                for w = 1:pyramids(b).wordCount
                    % Populate the bottom row with word features.

                    if hyperParams.embeddingTransformDepth > 0
                        pb.rawEmbeddings(:, b, w) = wordFeatures(:, pyramids(b).wordIndices(w));
                    else
                        pb.features(:, b, w, pb.N) = wordFeatures(:, pyramids(b).wordIndices(w));
                    end
                end
                pb.connectionLabels(b, 1:pyramids(b).wordCount - 1, pb.N - pyramids(b).wordCount + 1:pb.N - 1) = ...
                    pyramids(b).connectionLabels';
                pb.activeNode(b, 1:pyramids(b).wordCount, pb.N - pyramids(b).wordCount + 1:pb.N) = ...
                    pyramids(b).activeNode';
            end
        end
    end

    methods
        function [ topFeatures, connectionCosts ] = runForward(pb, embeddingTransformMatrix, connectionMatrix, compositionMatrix, hyperParams, trainingMode)

            connectionClassifierInputs = zeros((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + pb.NUMACTIONS, pb.B);

            % Run the optional embedding transformation layer forward.
            if ~isempty(embeddingTransformMatrix)
                for col = 1:pb.N
                    transformInputs = [ ones(1, pb.B); pb.rawEmbeddings(:, :, col) ];
                    pb.transformInnerActivations(:, :, col) = ...
                        embeddingTransformMatrix * transformInputs;
                    [ pb.features(:, :, col, pb.N), pb.masks(:, :, col) ] = ...
                        Dropout(tanh(pb.transformInnerActivations(:, :, col)), hyperParams.bottomDropout, trainingMode);
                end

                % Remove features for inactive nodes.
                pb.features(:, :, :, pb.N) = ...
                    bsxfun(@times, pb.features(:, :, :, pb.N) , ...
                           permute(pb.activeNode(:, :, pb.N), [3, 1, 2]));
            end

            connectionCosts = zeros(pb.B, 1);
            for row = pb.N - 1:-1:1
                for col = 1:row
                    % Compute the distribution over connections
                    connectionClassifierInputs = pb.collectConnectionClassifierInputs(hyperParams, col, row, connectionClassifierInputs);
                    [ pb.connections(:, :, col, row), localConnectionCosts ] = ...
                        ComputeSoftmaxLayer(connectionClassifierInputs, connectionMatrix, hyperParams, ...
                            pb.connectionLabels(:, col, row));
                    pb.connections(:, :, col, row) = bsxfun(@times, pb.connections(:, :, col, row), ...
                                                            permute(pb.activeNode(:, col, row), [2, 1, 3, 4]));

                    connectionCosts = connectionCosts + localConnectionCosts;

                    % Build the composed representation
                    compositionInputs = [ones(1, pb.B); pb.features(:, :, col, row + 1); pb.features(:, :, col + 1, row + 1)];
                    pb.compositionInnerActivations(:, :, col, row) = compositionMatrix * compositionInputs;
                    pb.compositionActivations(:, :, col, row) = tanh(pb.compositionInnerActivations(:, :, col, row));
                    % Multiply the three inputs by the three connection weights.
                    pb.features(:, :, col, row) = ...
                        bsxfun(@times, pb.features(:, :, col, row + 1), ...
                                       pb.connections(1, :, col, row)) + ...
                        bsxfun(@times, pb.features(:, :, col + 1, row + 1), ...
                                       pb.connections(2, :, col, row)) + ...
                        bsxfun(@times, pb.compositionActivations(:, :, col, row), ...
                                       pb.connections(3, :, col, row));
                end

                % Delete deltas for inactive nodes.
                features(:, :, :, row) = ...
                    bsxfun(@times, pb.features(:, :, :, row) , ...
                           permute(pb.activeNode(:, :, row), [3, 1, 2]));
            end

            % Collect features from the tops of each tree, not the top of the feature matrix.
            topFeatures = zeros(pb.D, pb.B);
            for b = 1:pb.B
                topFeatures(:, b) = pb.features(:, b, 1, pb.N - pb.wordCounts(b) + 1);
            end

            if ~trainingMode   
                % Temporary display method.
                % pb.connections(:,:,:,1)
                % pb.pyramids{1}.getText()
            end

            % Rescale the connection costs by the number of times supervision was applied.
            connectionCosts = connectionCosts ./ pb.numNodes;
        end

        function [ connectionClassifierInputs ] = collectConnectionClassifierInputs(pb, hyperParams, col, row, connectionClassifierInputs)
            contextPositions = -hyperParams.pyramidConnectionContextWidth + 1:hyperParams.pyramidConnectionContextWidth;

            for pos = 1:2 * hyperParams.pyramidConnectionContextWidth
                sourcePos = contextPositions(pos) + col;
                if (sourcePos > 0) && (sourcePos <= row + 1)
                    connectionClassifierInputs((pos - 1) * pb.D + 1:pos * pb.D, :) = pb.features(:, :, sourcePos, row + 1);
                else
                    connectionClassifierInputs((pos - 1) * pb.D + 1:pos * pb.D, :) = 0;
                end
                % Else: Leave in the zeros. Maybe fix replace with edge-of-sentence token? (TODO)
            end
            if col > 1
                connectionClassifierInputs(end - pb.NUMACTIONS + 1:end, :) = ...
                    pb.connections(:, :, col - 1, row);
            else
                connectionClassifierInputs(end - pb.NUMACTIONS + 1:end, :) = 0;
            end
        end

        function [ wordGradients, connectionMatrixGradients, ...
                   compositionMatrixGradients, embeddingTransformMatrixGradients ] = ...
            getGradient(pb, incomingDeltas, wordFeatures, embeddingTransformMatrix, connectionMatrix, compositionMatrix, hyperParams)
            % Run backwards.

            connectionClassifierInputs = zeros((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + pb.NUMACTIONS, pb.B);

            contextPositions = -hyperParams.pyramidConnectionContextWidth + 1:hyperParams.pyramidConnectionContextWidth;
            connectionMatrixGradients = zeros(size(connectionMatrix, 1), size(connectionMatrix, 2));
            compositionMatrixGradients = zeros(size(compositionMatrix, 1), size(compositionMatrix, 2));
            % Initialize delta matrix with the incoming deltas in the right places.

            % TODO: This could be represented as a vector covering only the deltas at one row,
            % but this could impose some time/complexity costs. Investigate.
            deltas = zeros(pb.D, pb.B, pb.N, pb.N);

            % Populate the delta matrix with the incoming deltas (reasonably fast).
            for b = 1:pb.B
                deltas(:, b, 1, pb.N - pb.wordCounts(b) + 1) = incomingDeltas(:, b);
            end

            % Initialize some variables that will be used inside the loop.
            deltasToConnections = zeros(pb.NUMACTIONS, pb.B);

            % Iterate over the structure in reverse
            for row = 1:pb.N - 1
                for col = row:-1:1
                    %% Handle composition function gradients %%

                    % Multiply in the three connection weights by the three inputs to the current features, 
                    % and add these to the existing deltas, which can either come from the inbound deltas (from the top-level classifier)
                    % or from a wide-window connection classifier.
                    deltas(:, :, col, row + 1) = ...
                        deltas(:, :, col, row + 1) + ...
                        bsxfun(@times, deltas(:, :, col, row), ...
                               pb.connections(1, :, col, row));
                    deltas(:, :, col + 1, row + 1) = ...
                        deltas(:, :, col + 1, row + 1) + ...
                        bsxfun(@times, deltas(:, :, col, row), ...
                               pb.connections(2, :, col, row));
                    compositionDeltas = ...
                        bsxfun(@times, deltas(:, :, col, row), ...
                               pb.connections(3, :, col, row));                 

                    % Backprop through the composition function.
                    [ localCompositionMatrixGradients, compositionDeltaLeft, compositionDeltaRight ] = ...
                        ComputeRNNLayerGradients(pb.features(:, :, col, row + 1), ...
                                                 pb.features(:, :, col + 1, row + 1), ...
                                                 compositionMatrix, compositionDeltas, @TanhDeriv, ...
                                                 pb.compositionInnerActivations(:, :, col, row));

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
                                                 pb.features(:, :, col, row + 1), 1);
                    deltasToConnections(2, :) = deltasToConnections(2, :) + ...
                                                 sum(deltas(:, :, col, row) .* ...
                                                 pb.features(:, :, col + 1, row + 1), 1);
                    deltasToConnections(3, :) = deltasToConnections(3, :) + ...
                                                 sum(deltas(:, :, col, row) .* ...
                                                 pb.compositionActivations(:, :, col, row), 1);
                    
                    % Compute gradients from the connection classifier wrt. the incoming deltas from above and to the right.
                    connectionClassifierInputs = pb.collectConnectionClassifierInputs(hyperParams, col, row, connectionClassifierInputs);
                    [ localConnectionMatrixGradients, connectionDeltas ] = ...
                        ComputeBareSoftmaxGradients(connectionMatrix, pb.connections(:, :, col, row), ...
                            deltasToConnections, connectionClassifierInputs);
                    connectionMatrixGradients = connectionMatrixGradients + localConnectionMatrixGradients;

                    % Compute gradients from the connection classifier wrt. the independent connection supervision signal.
                    [ localConnectionMatrixGradients, localConnectionDeltas ] = ...
                        ComputeSoftmaxClassificationGradients(connectionMatrix, pb.connections(:, :, col, row), ...
                            pb.connectionLabels(:, col, row), connectionClassifierInputs, hyperParams, pb.numNodes);
                    connectionMatrixGradients = connectionMatrixGradients + localConnectionMatrixGradients;

                    % Rescale by the number of times supervision is being applied.
                    connectionDeltas = connectionDeltas + localConnectionDeltas;

                    % Distribute the deltas from the softmax function back into its inputs.
                    for pos = 1:2 * hyperParams.pyramidConnectionContextWidth
                        sourcePos = contextPositions(pos) + col;
                        if sourcePos > 0 && sourcePos <= row + 1
                            deltas(:, :, sourcePos, row + 1) = ...
                                deltas(:, :, sourcePos, row + 1) + ...
                                connectionDeltas((pos - 1) * pb.D + 1:pos * pb.D, :);
                        end
                        if col > 1
                            deltasToConnections = bsxfun(@times, connectionDeltas(end + 1 - pb.NUMACTIONS:end, :), ...
                                                         permute(pb.activeNode(:, col, row), [2, 1, 3]));
                        else
                            deltasToConnections = zeros(pb.NUMACTIONS, pb.B);
                        end
                            
                    end
                end

                % Delete deltas for inactive nodes.
                deltas(:, :, :, row + 1) = ...
                       bsxfun(@times, deltas(:, :, :, row + 1) , ...
                           permute(pb.activeNode(:, :, row + 1), [3, 1, 2]));
            end

            % Run the embedding transform layers backwards.
            if hyperParams.embeddingTransformDepth > 0
                embeddingTransformMatrixGradients = zeros(size(embeddingTransformMatrix, 1), size(embeddingTransformMatrix, 2));                    rawEmbeddingDeltas = zeros(hyperParams.embeddingDim, pb.B, pb.N);
                rawEmbeddingDeltas = zeros(hyperParams.embeddingDim, pb.B, pb.N);
                for col = 1:pb.N
                    transformDeltas = deltas(:, :, col, pb.N) .* pb.masks(:, :, col); % Take dropout into account
                    [ localEmbeddingTransformMatrixGradients, rawEmbeddingDeltas(:, :, col) ] = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                              transformDeltas, pb.rawEmbeddings(:, :, col), ...
                              pb.transformInnerActivations(:, :, col), @TanhDeriv);
                    embeddingTransformMatrixGradients = embeddingTransformMatrixGradients + localEmbeddingTransformMatrixGradients;
                end
            else
                embeddingTransformMatrixGradients = [];
            end

            % Push deltas from the bottom row into the word gradients.
            % TODO: We don't need to require wordFeatures as an input, only used here.
            wordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), pb.N * pb.B);
            if hyperParams.trainWords
                for b = 1:pb.B
                    for col = 1:pb.wordCounts(b)
                        if hyperParams.embeddingTransformDepth > 0
                            wordGradients(:, pb.wordIndices(col, b)) = ...
                                wordGradients(:, pb.wordIndices(col, b)) + ...
                                rawEmbeddingDeltas(:, b, col);
                        else
                            wordGradients(:, pb.wordIndices(col, b)) = ...
                                wordGradients(:, pb.wordIndices(col, b)) + ...
                                deltas(:, b, col, pb.N);
                        end
                    end
                end
            end
        end
    end
end
