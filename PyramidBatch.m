% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef PyramidBatch < handle
    % Represents a single binary branching syntactic tree with three 
    % representations at each node:
    % - The index with which the feature vector can be looked up - if leaf
    % - The text with which the tree can be displayed.
    % - The features at the node.
    
    properties
        NUMACTIONS = 3;
        B = -1;
        D = -1;
        N = -1;
        wordIndices = [];
        wordCounts = [];
        features = [];
        compositionInnerActivations = [];
        compositionActivations = [];
        connections = [];
        connectionLabels = [];
        % TODO: Dropout
        % transformInnerActivations = []; % Stored activations for the embedding tranform layers. TODO.      
    end
    
    methods(Static)
        function pb = makePyramidBatch(pyramids, wordFeatures, hyperParams)
            pb = PyramidBatch();
            pb.B = length(pyramids);
            pb.D = hyperParams.dim;
            pb.N = 0;

            % Find the length of the longest sequence. We use this to set the size of the main feature matrix,
            % to this value has a large impact on the run time of the batch.
            for b = 1:pb.B
                pb.N = max([pb.N, pyramids.wordCount]);
            end

            pb.wordIndices = zeros(pb.N, pb.B);
            pb.wordCounts = zeros(pb.B, 1);
            pb.features = zeros(pb.N, pb.N * pb.D, pb.B);
            pb.compositionInnerActivations = zeros(pb.N - 1, (pb.N - 1) * pb.D, pb.B);
            pb.compositionActivations = zeros(pb.N - 1, (pb.N - 1) * pb.D, pb.B);
            pb.connections = zeros(pb.N - 1, pb.N - 1, pb.NUMACTIONS, pb.B);  % The above after softmax
            pb.connectionLabels = zeros(pb.N - 1, pb.N - 1, pb.B);

            for b = 1:pb.B
                pb.wordCounts(b) = pyramids(b).wordCount;
                pb.wordIndices(1:pb.wordCounts(b), b) = pyramids(b).wordIndices;
                for w = 1:pyramids(b).wordCount
                    % We assume there is no embedding transform layer, so just use the word features. (TODO)
                    pb.features(pb.N, (w - 1) * pb.D + 1:w * pb.D, b) = wordFeatures(pyramids(b).wordIndices(w), :)'; 
                end
                pb.connectionLabels(pb.N - pyramids(b).wordCount + 1:pb.N - 1, 1:pyramids(b).wordCount - 1, b) = ...
                    pyramids(b).connectionLabels;
            end 
        end

    end

    methods
        
        function [ topFeatures ] = runForward(pb, connectionMatrix, compositionMatrix, hyperParams)
            % TODO: Work out efficient assignment of rows/columns for main feature array.

            contextPositions = -hyperParams.pyramidConnectionContextWidth + 1:hyperParams.pyramidConnectionContextWidth;

            % Throwaway variable, storing would take up loads of space.
            connectionClassifierInputs = zeros((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + pb.NUMACTIONS, pb.B);

            for row = pb.N - 1:-1:1
                for col = 1:row

                    % TODO: Vectorize a bit more?
                    % Collect the inputs to the softmax function, first the features than the previous connections.
                    for pos = 1:2 * hyperParams.pyramidConnectionContextWidth
                        sourcePos = contextPositions(pos) + col;
                        if (sourcePos > 0) && (sourcePos <= row)
                            % I'll be shocked if there isn't an off-by-one somewhere in here.
                            connectionClassifierInputs((pos - 1) * pb.D + 1:pos * pb.D, :) = ...
                                pb.features(row + 1, (sourcePos - 1) * pb.D + 1:sourcePos * pb.D, :);
                        end
                        % Else: Leave in the zeros. Maybe fix replace with edge-of-sentence token?
                    end
                    if col > 1
                        connectionClassifierInputs((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + 1:end, :) = ...
                            pb.connections(row, col - 1, :, :);  % TODO: Need reshape?
                    end

                    % Compute the distribution over connections

                    % TODO: Make sure inputs vector is non-empty
                    pb.connections(row, col, :, :) = ComputeSoftmaxProbabilities(connectionClassifierInputs, connectionMatrix); % Reshape?

                    % Build the composed representation
                    compositionInputs = [ones(1, pb.B); permute(pb.features(row + 1, (col - 1) * pb.D + 1:(col + 1) * pb.D, :), [2, 3, 1])];
                    pb.compositionInnerActivations(row, (col - 1) * pb.D + 1:col * pb.D, :) = compositionMatrix * compositionInputs;
                    pb.compositionActivations(row, (col - 1) * pb.D + 1:col * pb.D, :) = tanh(pb.compositionInnerActivations(row, (col - 1) * pb.D + 1:col * pb.D, :));


                    % Multiply the three inputs by the three connection weights.
                    pb.features(row, (col - 1) * pb.D + 1:col * pb.D, :) = ...
                        bsxfun(@times, pb.features(row + 1, (col - 1) * pb.D + 1:col * pb.D, :), ...
                                       permute(pb.connections(row, col, 1, :), [1, 2, 4, 3])) + ...
                        bsxfun(@times, pb.features(row + 1, col * pb.D + 1:(col + 1) * pb.D, :), ...
                                       permute(pb.connections(row, col, 2, :), [1, 2, 4, 3])) + ...
                        bsxfun(@times, pb.compositionActivations(row, (col - 1) * pb.D + 1:col * pb.D, :), ...
                                       permute(pb.connections(row, col, 3, :), [1, 2, 4, 3]));
                end


            end

            % Collect features from the tops of each tree, not the top of the feature matrix.
            topFeatures = zeros(pb.D, pb.B);
            for b = 1:pb.B
                topFeatures(:, b) = pb.features(pb.N - pb.wordCounts(b) + 1, 1:pb.D, b);
            end
        end

        function [ wordGradients, connectionMatrixGradients, ...
                   compositionMatrixGradients, embeddingTransformMatrixGradients ] = ...
            getGradient(pb, incomingDeltas, ~, wordFeatures, connectionMatrix, ...
                        compositionMatrix, ~, ~, hyperParams)
            % Unused arguments are a relic to leave this compatible with non-batched ComputeCostAndGrad.

            contextPositions = -hyperParams.pyramidConnectionContextWidth + 1:hyperParams.pyramidConnectionContextWidth;

            connectionMatrixGradients = zeros(size(connectionMatrix, 1), size(connectionMatrix, 2));
            compositionMatrixGradients = zeros(size(compositionMatrix, 1), size(compositionMatrix, 2));

            % Initialize delta matrix with the incoming deltas in the right places
            % TODO: This could be represented as a vector covering only the deltas at one row,
            % but this could impose some time costs. Investigate.
            deltas = zeros(pb.N, pb.N * pb.D, pb.B);
            for b = 1:pb.B
                deltas(pb.N - pb.wordCounts(b) + 1, 1:pb.D, b) = incomingDeltas(:, b);
            end

            % Initialize some variables that will be used inside the loop.
            deltasToConnections = zeros(pb.NUMACTIONS, pb.B);
            connectionClassifierInputs = zeros((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + pb.NUMACTIONS, pb.B);

            % Iterate over the structure in reverse
            for row = 1:pb.N - 1
                for col = row:1

                    %% Handle composition gradients

                    % Multiply in the three connection weights by the three inputs to the current features.
                    deltas(row + 1, (col - 1) * pb.D + 1:col * pb.D, :) = ...
                        bsxfun(@times, deltas(row, (col - 1) * pb.D + 1:col * pb.D, :), ...
                               permute(pb.connections(row, col, 1, :), [1, 2, 4, 3]));
                    deltas(row + 1, col * pb.D + 1:(col + 1) * pb.D, :) = ...
                        bsxfun(@times, deltas(row, (col - 1) * pb.D + 1:col * pb.D, :), ...
                               permute(pb.connections(row, col, 2, :), [1, 2, 4, 3]));
                    compositionDeltas = ...
                        bsxfun(@times, deltas(row, (col - 1) * pb.D + 1:col * pb.D, :), ...
                               permute(pb.connections(row, col, 3, :), [1, 2, 4, 3]));                  

                    compositionDeltas = permute(compositionDeltas, [2, 3, 1]);  

                    % Backprop through the composition function.
                    [ localCompositionMatrixGradients, compositionDeltaLeft, compositionDeltaRight ] = ...
                        ComputeRNNLayerGradients(permute(pb.features(row + 1, (col - 1) * pb.D + 1:col * pb.D, :), [2, 3, 1]), ...
                                                 permute(pb.features(row + 1, col * pb.D + 1:(col + 1) * pb.D, :), [2, 3, 1]), ...
                                                 compositionMatrix, compositionDeltas, @TanhDeriv, ...
                                                 pb.compositionInnerActivations(row, (col - 1) * pb.D + 1:col * pb.D, :));

                    % Add the composition gradients into the gradient accumulators.
                    % TODO: If any non-zero gradients appear above the top of the smaller trees, they can be 
                    % summed in here. Make sure this doesn't happen.
                    compositionMatrixGradients = compositionMatrixGradients + sum(localCompositionMatrixGradients, 3);

                    deltas(row + 1, (col - 1) * pb.D + 1:col * pb.D, :) = ...
                        deltas(row + 1, (col - 1) * pb.D + 1:col * pb.D, :) + permute(compositionDeltaLeft, [3, 1, 2]);
                    deltas(row + 1, col * pb.D + 1:(col + 1) * pb.D, :) = ...
                        deltas(row + 1, col * pb.D + 1:(col + 1) * pb.D, :) + permute(compositionDeltaRight, [3, 1, 2]);

                    % Multiply the deltas by the three inputs to the current features to compute deltas for the connections.
                    deltasToConnections(1, :) = sum(deltas(row, (col - 1) * pb.D + 1:col * pb.D, :) .* ...
                                                 pb.features(row + 1, (col - 1) * pb.D + 1:col * pb.D, :));
                    deltasToConnections(2, :) = sum(deltas(row, (col - 1) * pb.D + 1:col * pb.D, :) .* ...
                                                 pb.features(row + 1, col * pb.D + 1:(col + 1) * pb.D, :));
                    deltasToConnections(1, :) = sum(deltas(row, (col - 1) * pb.D + 1:col * pb.D, :) .* ...
                                                 pb.compositionActivations(row, (col - 1) * pb.D + 1:col * pb.D, :));

                    % TODO: Vectorize a bit more?
                    % Collect the inputs to the softmax function, first the features than the previous connections.
                    for pos = 1:2 * hyperParams.pyramidConnectionContextWidth
                        sourcePos = contextPositions(pos) + col;
                        if (sourcePos > 0) && (sourcePos <= row)
                            % I'll be shocked if there isn't an off-by-one somewhere in here.
                            connectionClassifierInputs((pos - 1) * pb.D + 1:pos * pb.D, :) = ...
                                pb.features(row + 1, (sourcePos - 1) * pb.D + 1:sourcePos * pb.D, :);
                        end
                        % Else: Leave in the zeros. Maybe fix replace with edge-of-sentence token?
                    end
                    if col > 1
                        connectionClassifierInputs((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + 1:end, :) = ...
                            pb.connections(row, col - 1, :, :);
                    end

                    [ localConnectionMatrixGradients, connectionDeltas ] = ...
                        ComputeBareSoftmaxGradient(connectionMatrix, permute(pb.connections(row, col, :, :), [3, 4, 1, 2]), deltasToConnections, connectionClassifierInputs);

                    connectionMatrixGradients = connectionMatrixGradients + sum(localConnectionMatrixGradients, 3);

                    % TODO: Vectorize a bit more?
                    % Distribute the deltas from the softmax function back into its inputs.
                    for pos = 1:2 * hyperParams.pyramidConnectionContextWidth
                        sourcePos = contextPositions(pos) + col;
                        if (sourcePos > 0) && (sourcePos <= row)
                            % I'll be shocked if there isn't an off-by-one somewhere in here.
                            connectionDeltas((pos - 1) * pb.D + 1:pos * pb.D, :)
                            deltas(row + 1, (sourcePos - 1) * pb.D + 1:sourcePos * pb.D, :) = ...
                                deltas(row + 1, (sourcePos - 1) * pb.D + 1:sourcePos * pb.D, :) + ...
                                permute(connectionDeltas((pos - 1) * pb.D + 1:pos * pb.D, :), [3, 1, 2]);
                        end
                        % Else: Leave in the zeros. Maybe fix replace with edge-of-sentence token?
                        if col > 1
                            connectionClassifierInputs((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + 1:end, :) = ...
                                pb.connections(row, col - 1, :, :);
                        end
                    end

                    % TODO: Compute
                    embeddingTransformMatrixGradients = [];
                end
            end

            % Push gradients into words.
            wordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), pb.N * pb.B); 
            % TODO: We don't need to require wordFeatures outright, only used here.

            if hyperParams.trainWords
                for b = 1:pb.B
                    for col = 1:pb.wordCounts(b)
                        wordGradients(pb.wordIndices(col, b), :) = ...
                            wordGradients(pb.wordIndices(col, b), :) + ...
                            deltas(row, (col - 1) * pb.D + 1:col * pb.D, b);
                    end
                end
            end
        end
    end
end
