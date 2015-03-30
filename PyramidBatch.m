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
        wordIndices = [];  % The index into the embedding matrix for each word in the sequence.
        wordCounts = [];  % The number of words in each sequence.
        features = [];  % All computed activation vectors, with the words at the bottom row.
                        % All feature vectors on each row are concatenated, and can be separated
                        % with the help of colRng().
        compositionInnerActivations = [];  % Same structure as features, but contains the pre-nonlinearity
                                           % activations from the composition function, and has no bottom (word) layer.
        compositionActivations = [];  % Same structure as features, but contains the
                                      % activations from the composition function, and has no bottom (word) layer.
        connections = [];  % The length-3 vectors of weights for the three connection types at each position in the pyramid.
                           % Has no bottom (word) layer.
        connectionLabels = [];  % The optional correct connection type (in {1, 2, 3}) for each position in the pyramid.
        activeNode = [];  % Triangular boolean matrix for each batch entry indicating whether each position 
                          % is within the pyramid structure for that entry.

        % TODO: Add word-level dropout
        % TODO: Profile these functions and optimize where needed.
        % TODO: Move all summing over batch entries into the parent costgradfn to enable gradient clipping.
        % transformInnerActivations = []; % Stored activations for the embedding tranform layers. TODO.     
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

            pb.wordIndices = zeros(pb.N, pb.B);
            pb.wordCounts = [pyramids(:).wordCount];
            pb.features = zeros(pb.N, pb.N * pb.D, pb.B);
            pb.compositionInnerActivations = zeros(pb.N - 1, (pb.N - 1) * pb.D, pb.B);
            pb.compositionActivations = zeros(pb.N - 1, (pb.N - 1) * pb.D, pb.B);
            pb.connections = zeros(pb.N - 1, pb.N - 1, pb.NUMACTIONS, pb.B);
            pb.connectionLabels = zeros(pb.N - 1, pb.N - 1, pb.B);
            pb.activeNode = zeros(pb.N, pb.N, pb.B);

            % Copy data in from the individual batch entries.
            for b = 1:pb.B
                pb.wordIndices(1:pb.wordCounts(b), b) = pyramids(b).wordIndices;
                for w = 1:pyramids(b).wordCount
                    % We assume there is no embedding transform layer, so just use the word features. (TODO)
                    pb.features(pb.N, pb.colRng(w), b) = wordFeatures(pyramids(b).wordIndices(w), :)';
                end
                pb.connectionLabels(pb.N - pyramids(b).wordCount + 1:pb.N - 1, 1:pyramids(b).wordCount - 1, b) = ...
                    pyramids(b).connectionLabels;
                pb.activeNode(pb.N - pyramids(b).wordCount + 1:pb.N, 1:pyramids(b).wordCount, b) = ...
                    pyramids(b).activeNode;
            end
        end
    end

    methods
        function [ topFeatures, connectionCosts ] = runForward(pb, connectionMatrix, compositionMatrix, hyperParams)
            % TODO: Work out efficient assignment of rows/columns for main feature array.

            % TODO: Storing this variable btw forward and backward runs would take up loads of space, but could save time. Investigate.
            connectionClassifierInputs = zeros((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + pb.NUMACTIONS, pb.B);

            connectionCosts = zeros(pb.B, 1);

            for row = pb.N - 1:-1:1
                for col = 1:row
                    % Compute the distribution over connections
                    connectionClassifierInputs = pb.collectConnectionClassifierInputs(hyperParams, row, col);
                    [ pb.connections(row, col, :, :), localConnectionCosts ] = ...
                        ComputeSoftmaxLayer(connectionClassifierInputs, connectionMatrix, 1:pb.NUMACTIONS, pb.connectionLabels(row, col, :));
                    connectionCosts = connectionCosts + localConnectionCosts;

                    % Build the composed representation
                    compositionInputs = [ones(1, pb.B); permute(pb.features(row + 1, pb.colRng(col, col + 1), :), [2, 3, 1])];
                    pb.compositionInnerActivations(row, pb.colRng(col), :) = compositionMatrix * compositionInputs;
                    pb.compositionActivations(row, pb.colRng(col), :) = tanh(pb.compositionInnerActivations(row, pb.colRng(col), :));
                    % Multiply the three inputs by the three connection weights.
                    pb.features(row, pb.colRng(col), :) = ...
                        bsxfun(@times, pb.features(row + 1, pb.colRng(col), :), ...
                                       permute(pb.connections(row, col, 1, :), [1, 2, 4, 3])) + ...
                        bsxfun(@times, pb.features(row + 1, pb.colRng(col + 1), :), ...
                                       permute(pb.connections(row, col, 2, :), [1, 2, 4, 3])) + ...
                        bsxfun(@times, pb.compositionActivations(row, pb.colRng(col), :), ...
                                       permute(pb.connections(row, col, 3, :), [1, 2, 4, 3]));
                end
            end
            % Collect features from the tops of each tree, not the top of the feature matrix.
            topFeatures = zeros(pb.D, pb.B);
            for b = 1:pb.B
                topFeatures(:, b) = pb.features(pb.N - pb.wordCounts(b) + 1, 1:pb.D, b);
            end
        end

        function [ connectionClassifierInputs ] = collectConnectionClassifierInputs(pb, hyperParams, row, col)
            % TODO: Have this rewrite the variable in place?

            contextPositions = -hyperParams.pyramidConnectionContextWidth + 1:hyperParams.pyramidConnectionContextWidth;
            connectionClassifierInputs = zeros((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + pb.NUMACTIONS, pb.B);

            for pos = 1:2 * hyperParams.pyramidConnectionContextWidth
                sourcePos = contextPositions(pos) + col;
                if (sourcePos > 0) && (sourcePos <= row + 1)
                    connectionClassifierInputs(pb.colRng(pos), :) = ...
                        bsxfun(@times, pb.features(row + 1, pb.colRng(sourcePos), :), ...
                               pb.activeNode(row, col, :) .* pb.activeNode(row + 1, sourcePos, :));
                end
                % Else: Leave in the zeros. Maybe fix replace with edge-of-sentence token? (TODO)
            end
            if col > 1
                connectionClassifierInputs(end - pb.NUMACTIONS + 1:end, :) = ...
                    pb.connections(row, col - 1, :, :);
            end
        end

        function [ range ] = colRng(pb, startCol, endCol)
            % Computes the indices for the activations in feature matrix corresponding to columns start:end
            if nargin < 3
                % If no end is supplied, return the feature range for the single column startCol
                endCol = startCol;
            end
            range = (startCol - 1) * pb.D + 1:endCol * pb.D;
        end

        function [ wordGradients, connectionMatrixGradients, ...
                   compositionMatrixGradients, embeddingTransformMatrixGradients ] = ...
            getGradient(pb, incomingDeltas, ~, wordFeatures, connectionMatrix, compositionMatrix, ~, ~, hyperParams)
            % Run backwards.
            % Unused arguments are a relic to leave this compatible with non-batched ComputeCostAndGrad.

            contextPositions = -hyperParams.pyramidConnectionContextWidth + 1:hyperParams.pyramidConnectionContextWidth;
            connectionMatrixGradients = zeros(size(connectionMatrix, 1), size(connectionMatrix, 2));
            compositionMatrixGradients = zeros(size(compositionMatrix, 1), size(compositionMatrix, 2));
            % Initialize delta matrix with the incoming deltas in the right places.

            % TODO: This could be represented as a vector covering only the deltas at one row,
            % but this could impose some time/complexity costs. Investigate.
            deltas = zeros(pb.N, pb.N * pb.D, pb.B);

            % Populate the delta matrix with the incoming deltas.
            for b = 1:pb.B
                deltas(pb.N - pb.wordCounts(b) + 1, pb.colRng(1), b) = incomingDeltas(:, b);
            end

            % Initialize some variables that will be used inside the loop.
            deltasToConnections = zeros(pb.NUMACTIONS, pb.B);
            connectionClassifierInputs = zeros((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + pb.NUMACTIONS, pb.B);

            % Iterate over the structure in reverse
            for row = 1:pb.N - 1
                for col = row:-1:1
                    %% Handle composition function gradients %%

                    % Multiply in the three connection weights by the three inputs to the current features, 
                    % and add these to the existing deltas, which can either come from the inbound deltas (from the top-level classifier)
                    % or from a wide-window connection classifier.
                    deltas(row + 1, pb.colRng(col), :) = ...
                        deltas(row + 1, pb.colRng(col), :) + ...
                        bsxfun(@times, deltas(row, pb.colRng(col), :), ...
                               permute(pb.connections(row, col, 1, :), [1, 2, 4, 3]));
                    deltas(row + 1, pb.colRng(col + 1), :) = ...
                        deltas(row + 1, pb.colRng(col + 1), :) + ...
                        bsxfun(@times, deltas(row, pb.colRng(col), :), ...
                               permute(pb.connections(row, col, 2, :), [1, 2, 4, 3]));
                    compositionDeltas = ...
                        bsxfun(@times, deltas(row, pb.colRng(col), :), ...
                               permute(pb.connections(row, col, 3, :), [1, 2, 4, 3]));                 
                    compositionDeltas = permute(compositionDeltas, [2, 3, 1]); 

                    % Backprop through the composition function.
                    [ localCompositionMatrixGradients, compositionDeltaLeft, compositionDeltaRight ] = ...
                        ComputeRNNLayerGradients(permute(pb.features(row + 1, pb.colRng(col), :), [2, 3, 1]), ...
                                                 permute(pb.features(row + 1, pb.colRng(col + 1), :), [2, 3, 1]), ...
                                                 compositionMatrix, compositionDeltas, @TanhDeriv, ...
                                                 pb.compositionInnerActivations(row, pb.colRng(col), :));

                    % Add the composition gradients and deltas into the accumulators.
                    compositionMatrixGradients = compositionMatrixGradients + sum(localCompositionMatrixGradients, 3);
                    deltas(row + 1, pb.colRng(col), :) = ...
                        deltas(row + 1, pb.colRng(col), :) + permute(compositionDeltaLeft, [3, 1, 2]);
                    deltas(row + 1, pb.colRng(col + 1), :) = ...
                        deltas(row + 1, pb.colRng(col + 1), :) + permute(compositionDeltaRight, [3, 1, 2]);

                    %% Handle connection function gradients %%

                    % Multiply the deltas by the three inputs to the current features to compute deltas for the connections.
                    deltasToConnections(1, :) = deltasToConnections(1, :) + ...
                                                permute(sum(deltas(row, pb.colRng(col), :) .* ...
                                                 pb.features(row + 1, pb.colRng(col), :), 2), [1, 3, 2]);
                    deltasToConnections(2, :) = deltasToConnections(2, :) + ...
                                                 permute(sum(deltas(row, pb.colRng(col), :) .* ...
                                                 pb.features(row + 1, pb.colRng(col + 1), :), 2), [1, 3, 2]);
                    deltasToConnections(3, :) = deltasToConnections(3, :) + ...
                                                 permute(sum(deltas(row, pb.colRng(col), :) .* ...
                                                 pb.compositionActivations(row, pb.colRng(col), :), 2), [1, 3, 2]);
                    
                    % Compute gradients from the connection classifier wrt. the incoming deltas from above and to the right.
                    connectionClassifierInputs = pb.collectConnectionClassifierInputs(hyperParams, row, col);
                    [ localConnectionMatrixGradients, connectionDeltas ] = ...
                        ComputeBareSoftmaxGradients(connectionMatrix, permute(pb.connections(row, col, :, :), [3, 4, 1, 2]), ...
                            deltasToConnections, connectionClassifierInputs);
                    connectionMatrixGradients = connectionMatrixGradients + sum(localConnectionMatrixGradients, 3);

                    % Compute gradients from the connection classifier wrt. the independent connection supervision signal.
                    [ localConnectionMatrixGradients, localConnectionDeltas ] = ...
                        ComputeSoftmaxClassificationGradients(connectionMatrix, permute(pb.connections(row, col, :, :), [3, 4, 1, 2]), ...
                            pb.connectionLabels(row, col, :), connectionClassifierInputs);
                    connectionMatrixGradients = connectionMatrixGradients + sum(localConnectionMatrixGradients, 3);
                    connectionDeltas = connectionDeltas + localConnectionDeltas;

                    % Distribute the deltas from the softmax function back into its inputs.
                    for pos = 1:2 * hyperParams.pyramidConnectionContextWidth
                        sourcePos = contextPositions(pos) + col;
                        if sourcePos > 0 && sourcePos <= row + 1
                            % Multiply in the activeNode units for both the source and destination for the deltas
                            % to avoid propagating through nonexistent connections.
                            deltas(row + 1, pb.colRng(sourcePos), :) = ...
                                deltas(row + 1, pb.colRng(sourcePos), :) + ...
                                bsxfun(@times, permute(connectionDeltas(pb.colRng(pos), :), [3, 1, 2]), ...
                                               pb.activeNode(row, col, :) .* pb.activeNode(row + 1, sourcePos, :));
                        end
                        if col > 1
                            deltasToConnections = bsxfun(@times, connectionDeltas(end + 1 - pb.NUMACTIONS:end, :), ...
                                                         permute(pb.activeNode(row, col, :), [1, 3, 2]));
                        else
                            deltasToConnections = zeros(pb.NUMACTIONS, pb.B);
                        end
                            
                    end
                end
            end

            % Push deltas from the bottom row into the word gradients.
            % TODO: We don't need to require wordFeatures as an input, only used here.
            wordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), pb.N * pb.B);
            if hyperParams.trainWords
                for b = 1:pb.B
                    for col = 1:pb.wordCounts(b)
                        wordGradients(pb.wordIndices(col, b), :) = ...
                            wordGradients(pb.wordIndices(col, b), :) + ...
                            deltas(pb.N, pb.colRng(col), b);
                    end
                end
            end

            % TODO: Reintroduce this feature.
            embeddingTransformMatrixGradients = [];
        end
    end
end
