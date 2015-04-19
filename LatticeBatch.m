% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef LatticeBatch < handle
   
    properties
        NUMACTIONS = 3;  % The number of connection types.
                         % 1 := Copy left child.
                         % 2 := Copy right child.
                         % 3 := Compose left and right children.
        B = -1;  % Batch size.
        D = -1;  % Number of dimensions in the feature vector at each position.
        N = -1;  % The length of the longest sequence that can be handled within this lattice.
        lattices = [];  % The lattice objects in the batch.
        wordIndices = [];  % The index into the embedding matrix for each word in the sequence.
        wordCounts = [];  % The number of words in each sequence.
        features = [];  % All computed activation vectors, with the words at the bottom row.
                        % For speed, indexing is by (dim, batch-entry, column, row)
        rawEmbeddings = [];  % Word embeddings. Used only in conjunction with an embedding transform layer.
        masks = [];  % Same structure as the bottom row of features, but contains dropout masks for the embedding transform layer.
        compositionActivations = [];  % Same structure as features, but contains the
                                      % activations from the composition function, and has no bottom (word) layer.
        scores = [];  % The inputs to the connection softmax at each row.
        connections = [];  % The length-3 vectors of weights for the three connection types at each position in the lattice.
                           % Has no bottom (word) layer.
        connectionLabels = [];  % The optional correct connection type (in {1, 2, 3}) for each position in the lattice.
        activeNode = [];  % Triangular boolean matrix for each batch entry indicating whether each position 
                          % is within the lattice structure for that entry.
    end
   
    methods(Static)
        function lb = makeLatticeBatch(lattices, wordFeatures, hyperParams)
            % Constructor: create and populate the batch data structures using a specific batch of data.
            % NOTE: This class is designed for use in a typical SGD setting (as in TrainSGD here) where batches are created, used once
            % and then destroyed. As such, this constructor bakes certain learned model parameters into the batch
            % object, and this any object created this way will become stale after one gradient step.
            lb = LatticeBatch();
            lb.B = length(lattices);
            lb.D = hyperParams.dim;

            % Find the length of the longest sequence. We use this to set the size of the main feature matrix,
            % to this value has a large impact on the run time of the batch.
            lb.N = max([lattices(:).wordCount]);

            lb.lattices = cell(lb.B, 1);

            lb.wordIndices = zeros(lb.N, lb.B);
            lb.wordCounts = [lattices(:).wordCount];
            lb.features = zeros(lb.D, lb.B, lb.N, lb.N);
            lb.rawEmbeddings = zeros(hyperParams.embeddingDim, lb.B, hyperParams.embeddingTransformDepth * lb.N);
            lb.masks = zeros(lb.D, lb.B, hyperParams.embeddingTransformDepth * lb.N);
            lb.compositionActivations = zeros(lb.D, lb.B, lb.N - 1, lb.N - 1);
            lb.scores = zeros(lb.N - 1, lb.B, lb.N - 1);
            lb.connections = zeros(lb.NUMACTIONS, lb.B, lb.N - 1, lb.N - 1);
            lb.connectionLabels = zeros(lb.B, lb.N - 1); %% deleted last
            lb.activeNode = zeros(lb.B, lb.N, lb.N);

            % Copy data in from the individual batch entries.
            for b = 1:lb.B
                lb.lattices{b} = lattices(b);
                lb.wordIndices(1:lb.wordCounts(b), b) = lattices(b).wordIndices;
                for w = 1:lattices(b).wordCount
                    % Populate the bottom row with word features.

                    if hyperParams.embeddingTransformDepth > 0
                        lb.rawEmbeddings(:, b, w) = wordFeatures(:, lattices(b).wordIndices(w));
                    else
                        lb.features(:, b, w, lb.N) = wordFeatures(:, lattices(b).wordIndices(w));
                    end
                end
                lb.connectionLabels(b, lb.N - lattices(b).wordCount + 1:lb.N - 1) = ...
                    lattices(b).connectionLabels';
                lb.activeNode(b, 1:lattices(b).wordCount, lb.N - lattices(b).wordCount + 1:lb.N) = ...
                    lattices(b).activeNode';
            end
        end
    end

    methods
        function [ topFeatures, connectionCosts, connectionAccuracy ] = ...
            runForward(lb, embeddingTransformMatrix, connectionMatrix, compositionMatrix, hyperParams, trainingMode)

            % Run the optional embedding transformation layer forward.
            if ~isempty(embeddingTransformMatrix)
                for col = 1:lb.N
                    transformInputs = [ ones(1, lb.B); lb.rawEmbeddings(:, :, col) ];
                    [ lb.features(:, :, col, lb.N), lb.masks(:, :, col) ] = ...
                        Dropout(tanh(embeddingTransformMatrix * transformInputs), hyperParams.bottomDropout, trainingMode);
                end

                % Remove features for inactive nodes.
                lb.features(:, :, :, lb.N) = ...
                    bsxfun(@times, lb.features(:, :, :, lb.N) , ...
                           permute(lb.activeNode(:, :, lb.N), [3, 1, 2]));
            end

            % Prepare to compute connection accuracy.
            if hyperParams.showDetailedStats
                correctConnectionLabels = 0;
                totalConnectionLabels = sum(lb.wordCounts - 2);
            end

            connectionCosts = zeros(lb.B, 1);
            contextPositions = -hyperParams.latticeConnectionContextWidth + 1:hyperParams.latticeConnectionContextWidth;
            for row = lb.N - 1:-1:1

                % Decide where to compose.
                if row > 1  % Choosing the best of one is meaningless.
                    % Precompute a padded version of the features.
                    paddedRow = padarray(lb.features(:, :, :, row + 1), [1, 0, hyperParams.latticeConnectionContextWidth - 1]);

                    % We only need to pad at the bottom, to handle the column index features.
                    paddedRow = paddedRow(2:end, :, :);

                    for col = 1:row
                        % Score each node in the row.
                        scorerInputs = paddedRow(:, :, contextPositions + col + hyperParams.latticeConnectionContextWidth - 1);
                        scorerInputs(end, :, 1) = col;
                        scorerInputs(end, :, 2) = (1.0 .* col) / row;
                        lb.scores(col, :, row) = sum(sum(bsxfun(@times, permute(connectionMatrix, [1, 3, 2]), scorerInputs), 1), 3);
                    end

                    % Softmax the scores.
                    [ merges, localConnectionCosts ] = ...
                        ComputeSoftmaxLayer(lb.scores(1:row, :, row), [], hyperParams, lb.connectionLabels(:, row), lb.activeNode(:, 1:row, row)');
                    connectionCosts = connectionCosts + localConnectionCosts;

                    % Zero out 0/0s from inactive nodes.
                    merges(isnan(merges)) = 0; 

                    lb.connections(3, :, 1:row, row) = merges';

                    if hyperParams.showDetailedStats
                        [ ~, preds ] = max(lb.connections(3, :, :, row), [], 3);
                        correctConnectionLabels = correctConnectionLabels + sum(preds' == lb.connectionLabels(:, row));
                    end
                else
                    lb.connections(3, :, 1, row) = 1;
                end

                % TODO: Parfor?
                for col = 1:row
                    % Compute the rest of the connection weights.
                    lb.connections(1, :, col, row) = sum(lb.connections(3, :, col + 1:row, row), 3);
                    lb.connections(2, :, col, row) = sum(lb.connections(3, :, 1:col - 1, row), 3);

                    % Build the composed representation
                    compositionInputs = [ones(1, lb.B); lb.features(:, :, col, row + 1); lb.features(:, :, col + 1, row + 1)];
                    lb.compositionActivations(:, :, col, row) = tanh(compositionMatrix * compositionInputs);

                    % Multiply the three inputs by the three connection weights.
                    % NOTE: This is a major bottleneck. Keep an eye out for possible speedups.
                    lb.features(:, :, col, row) = ...
                        bsxfun(@times, lb.features(:, :, col, row + 1), ...
                                       lb.connections(1, :, col, row)) + ...
                        bsxfun(@times, lb.features(:, :, col + 1, row + 1), ...
                                       lb.connections(2, :, col, row)) + ...
                        bsxfun(@times, lb.compositionActivations(:, :, col, row), ...
                                       lb.connections(3, :, col, row));
                end
            end

            % Collect features from the tops of each tree, not the top of the feature matrix.
            topFeatures = zeros(lb.D, lb.B);
            for b = 1:lb.B
                topFeatures(:, b) = lb.features(:, b, 1, lb.N - lb.wordCounts(b) + 1);
            end

            % Rescale the connection costs by the number of times supervision was applied.
            connectionCosts = (connectionCosts ./ (lb.wordCounts' - 2)) .* hyperParams.connectionCostScale;

            % Zero out 0/0s from inactive nodes.
            connectionCosts(isnan(connectionCosts)) = 0;

            if ~trainingMode   
                % Temporary display method.
                % lb.connections(:,:,:,1)
                % lb.lattices{1}.getText()
            end

            if hyperParams.showDetailedStats
                connectionAccuracy = correctConnectionLabels / totalConnectionLabels;
            else
                connectionAccuracy = -1;
            end
        end

        function [ wordGradients, connectionMatrixGradients, ...
                   compositionMatrixGradients, embeddingTransformMatrixGradients ] = ...
            getGradient(lb, incomingDeltas, wordFeatures, embeddingTransformMatrix, connectionMatrix, compositionMatrix, hyperParams)
            % Run backwards.

            contextPositions = -hyperParams.latticeConnectionContextWidth + 1:hyperParams.latticeConnectionContextWidth;
            connectionMatrixGradients = zeros(size(connectionMatrix, 1), size(connectionMatrix, 2));
            compositionMatrixGradients = zeros(size(compositionMatrix, 1), size(compositionMatrix, 2));
            % Initialize delta matrix with the incoming deltas in the right places.

            % TODO: This could be represented as a vector covering only the deltas at one row,
            % but this could impose some time/complexity costs. Investigate.
            deltas = zeros(lb.D, lb.B, lb.N, lb.N);

            % Populate the delta matrix with the incoming deltas (reasonably fast).
            for b = 1:lb.B
                deltas(:, b, 1, lb.N - lb.wordCounts(b) + 1) = incomingDeltas(:, b);
            end

            % Iterate over the structure in reverse
            for row = 1:lb.N - 1
                % Get the score deltas from the softmax.
               
                % Push these deltas into the softmax that predicts where to merge.
                deltasToMerges = zeros(row, lb.B);

                for col = 1:row
                    %% Handle composition function gradients %%

                    % Multiply in the three connection weights by the three inputs to the current features, 
                    % and add these to the existing deltas, which can either come from the inbound deltas (from the top-level classifier)
                    % or from a wide-window connection classifier.
                    deltas(:, :, col, row + 1) = ...
                        deltas(:, :, col, row + 1) + ...
                        bsxfun(@times, deltas(:, :, col, row), ...
                               lb.connections(1, :, col, row));
                    deltas(:, :, col + 1, row + 1) = ...
                        deltas(:, :, col + 1, row + 1) + ...
                        bsxfun(@times, deltas(:, :, col, row), ...
                               lb.connections(2, :, col, row));
                    compositionDeltas = ...
                        bsxfun(@times, deltas(:, :, col, row), ...
                               lb.connections(3, :, col, row));                 

                    % Backprop through the composition function.
                    [ localCompositionMatrixGradients, compositionDeltaLeft, compositionDeltaRight ] = ...
                        ComputeRNNLayerGradients(lb.features(:, :, col, row + 1), ...
                                                 lb.features(:, :, col + 1, row + 1), ...
                                                 compositionMatrix, compositionDeltas, @TanhDeriv, ...
                                                 lb.compositionActivations(:, :, col, row));

                    % Add the composition gradients and deltas into the accumulators.
                    compositionMatrixGradients = compositionMatrixGradients + localCompositionMatrixGradients;
                    deltas(:, :, col, row + 1) = ...
                        deltas(:, :, col, row + 1) + compositionDeltaLeft;
                    deltas(:, :, col + 1, row + 1) = ...
                        deltas(:, :, col + 1, row + 1) + compositionDeltaRight;

                    %% Handle connection function gradients %%

                    if row > 1
                        % Multiply the deltas by the three inputs to the current features to compute deltas for the connections.
                        deltasToMerges(col + 1:row, :) = bsxfun(@plus, deltasToMerges(col + 1:row, :), ...
                                                          sum(deltas(:, :, col, row) .* ...
                                                              lb.features(:, :, col, row + 1), 1));

                        deltasToMerges(1:col - 1, :) =  bsxfun(@plus, deltasToMerges(1:col - 1, :), ...
                                                        sum(deltas(:, :, col, row) .* ...
                                                            lb.features(:, :, col + 1, row + 1), 1));

                        deltasToMerges(col, :) = bsxfun(@plus, deltasToMerges(col, :), ... 
                                                    sum(deltas(:, :, col, row) .* ...
                                                     lb.compositionActivations(:, :, col, row), 1));
                    end
                end

                if row > 1
                    % deltasToMerges(:, :) = bsxfun(@times, deltasToMerges(:, :), lb.activeNode(:, 1:lb.N - 1, row)');

                    merges = permute(lb.connections(3, :, 1:row, row), [3, 2, 1, 4]);

                    % Compute gradients for the scores wrt. the incoming deltas from above and to the right.
                    [ ~, incomingDeltasToScores ] = ...
                        ComputeBareSoftmaxGradients([], merges, deltasToMerges, lb.scores(1:row, :, row));

                    % Compute gradients for the scores wrt. the independent connection supervision signal.
                    [ ~, labelDeltasToScores ] = ...
                            ComputeSoftmaxClassificationGradients([], merges, lb.connectionLabels(:, row), ...
                                lb.scores(1:row, :, row), hyperParams, (lb.wordCounts - 2)' ./ hyperParams.connectionCostScale);

                    deltasToScores = labelDeltasToScores + incomingDeltasToScores;

                    % Overwrite 0/0 deltas from inactive nodes.
                    deltasToScores(isnan(deltasToScores)) = 0;

                    % Precompute a padded version of the features.
                    paddedRow = padarray(lb.features(:, :, :, row + 1), [1, 0, hyperParams.latticeConnectionContextWidth - 1]);

                    % We only need to pad at the bottom, to handle the column index features.
                    paddedRow = paddedRow(2:end, :, :);

                    for col = 1:row
                        scorerInputs = paddedRow(:, :, contextPositions + col + hyperParams.latticeConnectionContextWidth - 1);
                        scorerInputs(end, :, 1) = col;
                        scorerInputs(end, :, 2) = (1.0 .* col) / row;
                        deltasToContext = bsxfun(@times, permute(connectionMatrix, [1, 3, 2]), deltasToScores(col, :));
                        connectionMatrixGradients = connectionMatrixGradients + ...
                                permute(sum(bsxfun(@times, scorerInputs, deltasToScores(col, :)), 2), [1, 3, 2]);

                        % Distribute the deltas from the softmax function back into its inputs.
                        for pos = 1:2 * hyperParams.latticeConnectionContextWidth
                            sourcePos = contextPositions(pos) + col;
                            if sourcePos > 0 && sourcePos <= row + 1
                                deltas(:, :, sourcePos, row + 1) = ...
                                    deltas(:, :, sourcePos, row + 1) + ...
                                    deltasToContext(1:end - 1, :, pos);
                            end 
                        end
                    end
                end

                % Scale down the deltas.
                if hyperParams.maxDeltaNorm > 0
                    multipliers = min(bsxfun(@rdivide, hyperParams.maxDeltaNorm, sum(deltas(:, :, :, row + 1).^2)), ones(1, lb.B, lb.N));
                    deltas(:, :, :, row + 1) = bsxfun(@times, deltas(:, :, :, row + 1), multipliers);
                end
            end

            % Run the embedding transform layers backwards.
            if hyperParams.embeddingTransformDepth > 0
                embeddingTransformMatrixGradients = zeros(size(embeddingTransformMatrix, 1), size(embeddingTransformMatrix, 2));                    
                rawEmbeddingDeltas = zeros(hyperParams.embeddingDim, lb.B, lb.N);
                for col = 1:lb.N
                    transformDeltas = deltas(:, :, col, lb.N) .* lb.masks(:, :, col); % Take dropout into account
                    [ localEmbeddingTransformMatrixGradients, rawEmbeddingDeltas(:, :, col) ] = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                              transformDeltas, lb.rawEmbeddings(:, :, col), ...
                              lb.features(:, :, col, lb.N), @TanhDeriv);
                    embeddingTransformMatrixGradients = embeddingTransformMatrixGradients + localEmbeddingTransformMatrixGradients;
                end
            else
                embeddingTransformMatrixGradients = [];
            end

            % Push deltas from the bottom row into the word gradients.
            % TODO: We don't need to require wordFeatures as an input, only used here.
            wordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), lb.N * lb.B);
            if hyperParams.trainWords
                for b = 1:lb.B
                    for col = 1:lb.wordCounts(b)
                        if hyperParams.embeddingTransformDepth > 0
                            wordGradients(:, lb.wordIndices(col, b)) = ...
                                wordGradients(:, lb.wordIndices(col, b)) + ...
                                rawEmbeddingDeltas(:, b, col);
                        else
                            wordGradients(:, lb.wordIndices(col, b)) = ...
                                wordGradients(:, lb.wordIndices(col, b)) + ...
                                deltas(:, b, col, lb.N);
                        end
                    end
                end
            end
        end
    end
end
