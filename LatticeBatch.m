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
        LSTM_IFOGf = []  % Intermediate representations used only with LSTM activations.
        scorerHiddenLayer = [];  % The activations inside the scorer before the linear combination.
        scores = [];  % The inputs to the connection softmax at each row.
        connections = [];  % The length-3 vectors of weights for the three connection types at each position in the lattice.
                           % Has no bottom (word) layer.
        connectionLabels = [];  % The optional correct connection type (in {1, 2, 3}) for each position in the lattice.
        activeNode = [];  % Triangular boolean matrix for each batch entry indicating whether each position 
                          % is within the lattice structure for that entry.
        supervisionWeights = [];  % Multiplicative weighting factors to apply to the cost/gradient for connections.

        rawLeftEdgeEmbedding = [];   % The untransformed versions of the extra word embeddings that are fed to the 
        rawRightEdgeEmbedding = [];  % scorer at the edges of each sentence.
        leftEdgeEmbedding = [];   % The transformed verisions of the above.
        rightEdgeEmbedding = [];

        % If LSTM composition is used, an additional dimension is used to distinguish hidden state (1) and cell state (2).
        % TODO: Store IFOG for LSTM.
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
            lb.features = zeros(lb.D, lb.B, lb.N, lb.N, hyperParams.lstm + 1);
            lb.rawEmbeddings = zeros(hyperParams.embeddingDim, lb.B, hyperParams.embeddingTransformDepth * lb.N);
            lb.masks = zeros(lb.D, lb.B, hyperParams.embeddingTransformDepth * (lb.N + 2));
            lb.compositionActivations = zeros(lb.D, lb.B, lb.N - 1, lb.N - 1, hyperParams.lstm + 1);
            lb.LSTM_IFOGf = zeros(lb.D * 5, lb.B, lb.N - 1, lb.N - 1, hyperParams.lstm);
            lb.scorerHiddenLayer = zeros(hyperParams.latticeConnectionHiddenDim, lb.B, lb.N, lb.N);
            lb.scores = zeros(lb.N - 1, lb.B, lb.N - 1);
            lb.connections = zeros(lb.NUMACTIONS, lb.B, lb.N - 1, lb.N - 1);
            lb.connectionLabels = zeros(lb.B, lb.N - 1); %% deleted last
            lb.activeNode = zeros(lb.B, lb.N, lb.N);
            lb.supervisionWeights = ones(lb.B, lb.N);
            lb.rawLeftEdgeEmbedding = wordFeatures(:, hyperParams.sentenceStartWordIndex);
            lb.rawRightEdgeEmbedding = wordFeatures(:, hyperParams.sentenceEndWordIndex);
            lb.leftEdgeEmbedding = zeros(lb.D, 1);
            lb.rightEdgeEmbedding = zeros(lb.D, 1);

            % Copy data in from the individual batch entries.
            for b = 1:lb.B
                lb.lattices{b} = lattices(b);
                lb.wordIndices(1:lb.wordCounts(b), b) = lattices(b).wordIndices;
                for w = 1:lattices(b).wordCount
                    % Populate the bottom row with word features.

                    if hyperParams.embeddingTransformDepth > 0
                        lb.rawEmbeddings(:, b, w) = wordFeatures(:, lattices(b).wordIndices(w));
                    else
                        lb.features(:, b, w, lb.N, 1) = wordFeatures(:, lattices(b).wordIndices(w));
                    end
                end
                lb.connectionLabels(b, lb.N - lattices(b).wordCount + 1:lb.N - 1) = ...
                    lattices(b).connectionLabels';
                lb.activeNode(b, 1:lattices(b).wordCount, lb.N - lattices(b).wordCount + 1:lb.N) = ...
                    lattices(b).activeNode';
            end

            assert(~hyperParams.lstm || sum(sum(sum(sum(lb.features(:,:,:,:,2))))) == 0);
        end
    end

    methods
        function [ topFeatures, connectionCosts, connectionAccuracy ] = ...
            runForward(lb, embeddingTransformMatrix, connectionMatrix, scoringVector, compositionMatrix, hyperParams, trainingMode)

            % Run the optional embedding transformation layer forward.
            if ~isempty(embeddingTransformMatrix)
                for col = 1:lb.N
                    transformInputs = [ ones(1, lb.B); lb.rawEmbeddings(:, :, col) ];
                    [ lb.features(:, :, col, lb.N, 1), lb.masks(:, :, col) ] = ...
                        Dropout(tanh(embeddingTransformMatrix * transformInputs), hyperParams.bottomDropout, trainingMode);
                end

                % Remove features for inactive nodes.
                lb.features(:, :, :, lb.N, :) = ...
                    bsxfun(@times, lb.features(:, :, :, lb.N, :) , ...
                           permute(lb.activeNode(:, :, lb.N), [3, 1, 2]));

                [ lb.leftEdgeEmbedding, lb.masks(:, 1, lb.N + 1) ] = ...
                    Dropout(tanh(embeddingTransformMatrix * [1; lb.rawLeftEdgeEmbedding]), hyperParams.bottomDropout, trainingMode);
                [ lb.rightEdgeEmbedding, lb.masks(:, 1, lb.N + 2) ] = ...
                    Dropout(tanh(embeddingTransformMatrix * [1; lb.rawRightEdgeEmbedding]), hyperParams.bottomDropout, trainingMode);
            else
                lb.leftEdgeEmbedding = lb.rawLeftEdgeEmbedding;
                lb.rightEdgeEmbedding = lb.rawRightEdgeEmbedding;
            end

            % Prepare to compute connection accuracy.
            if hyperParams.showDetailedStats
                correctConnectionLabels = zeros(lb.B, 1);
                totalConnectionLabels = max(0, lb.wordCounts' - 2);
            end

            connectionCosts = zeros(lb.B, 1);
            contextPositions = -hyperParams.latticeConnectionContextWidth + 1:hyperParams.latticeConnectionContextWidth;
            for row = lb.N - 1:-1:1

                % Decide where to compose.
                if row > 1  % Choosing the best of one is meaningless.
                    % Precompute a padded version of the features.
                    % Dimensions: D, col, B, row
                    paddedRow = permute(padarray(lb.features(:, :, :, row + 1, 1), [0, 0, hyperParams.latticeConnectionContextWidth - 1]), [1, 3, 2, 4, 5]);

                    % Apply <s>/</s>
                    paddedRow(:, hyperParams.latticeConnectionContextWidth - 1, :) = ...
                        repmat(lb.leftEdgeEmbedding, [1, 1, lb.B]);
                    paddedRow(:, sub2ind([lb.N + (2 * (hyperParams.latticeConnectionContextWidth - 1)), lb.B], ...
                        max(1, hyperParams.latticeConnectionContextWidth + lb.wordCounts + row + 1 - lb.N), 1:lb.B)) = ...
                        repmat(lb.rightEdgeEmbedding, [1, 1, lb.B]);

                    for col = 1:row
                        % Score each node in the row with a two-layer network with a single output unit.

                        scorerHiddenInputs = [ones(3, lb.B); ...
                            reshape(paddedRow(:, contextPositions + col + hyperParams.latticeConnectionContextWidth - 1, :), ...
                            [2 * lb.D * hyperParams.latticeConnectionContextWidth, lb.B])];
                        scorerHiddenInputs(2, :) = col;
                        scorerHiddenInputs(3, :) = (1.0 .* col) / row;

                        % Dimensions: hiddenD x B
                        lb.scorerHiddenLayer(:, :, col, row) = tanh(connectionMatrix * scorerHiddenInputs);
                        lb.scores(col, :, row) = scoringVector * [ones(1, lb.B); lb.scorerHiddenLayer(:, :, col, row)];
                    end

                    % Softmax the scores.
                    [ merges, localConnectionCosts, probCorrect ] = ...
                        ComputeSoftmaxLayer(lb.scores(1:row, :, row), [], hyperParams, lb.connectionLabels(:, row), hyperParams.connectionCostScale ./ (lb.wordCounts' - 2), lb.activeNode(:, 1:row, row)');

                    lb.connections(3, :, 1:row, row) = merges';

                    if hyperParams.latticeLocalCurriculum
                        connectionCosts = connectionCosts .* lb.supervisionWeights(:, row);
                        lb.supervisionWeights(:, row - 1) = probCorrect .* lb.supervisionWeights(:, row);
                    end

                    connectionCosts = connectionCosts + localConnectionCosts;

                    if hyperParams.showDetailedStats
                        [ ~, preds ] = max(lb.connections(3, :, :, row), [], 3);
                        correctConnectionLabels = correctConnectionLabels + (preds' == lb.connectionLabels(:, row) .* lb.activeNode(:, 2, row));
                    end
                else
                    lb.connections(3, :, 1, row) = 1;
                end

                for col = 1:row
                    % Compute the rest of the connection weights.
                    lb.connections(1, :, col, row) = sum(lb.connections(3, :, col + 1:row, row), 3);
                    lb.connections(2, :, col, row) = sum(lb.connections(3, :, 1:col - 1, row), 3);

                    % Build the composed representation
                    if hyperParams.lstm
                        [ lb.compositionActivations(:, :, col, row, 1), lb.compositionActivations(:, :, col, row, 2), ...
                            lb.LSTM_IFOGf(:, :, col, row) ] = ...
                            ComputeTreeLSTMLayer(compositionMatrix, lb.features(:, :, col, row + 1, 1), lb.features(:, :, col + 1, row + 1, 1), ...
                                             lb.features(:, :, col, row + 1, 2), lb.features(:, :, col + 1, row + 1, 2));
                    else
                        compositionInputs = [ones(1, lb.B); lb.features(:, :, col, row + 1, 1); lb.features(:, :, col + 1, row + 1, 1)];
                        lb.compositionActivations(:, :, col, row, 1) = tanh(compositionMatrix * compositionInputs);
                    end

                    % Multiply the three inputs by the three connection weights.
                    % NOTE: This is a major bottleneck. Keep an eye out for possible speedups.
                    lb.features(:, :, col, row, :) = ...
                        bsxfun(@times, lb.features(:, :, col, row + 1, :), ...
                                       lb.connections(1, :, col, row)) + ...
                        bsxfun(@times, lb.features(:, :, col + 1, row + 1, :), ...
                                       lb.connections(2, :, col, row)) + ...
                        bsxfun(@times, lb.compositionActivations(:, :, col, row, :), ...
                                       lb.connections(3, :, col, row));
                end

                % Remove features for inactive nodes.
                lb.features(:, :, :, row, :) = ...
                    bsxfun(@times, lb.features(:, :, :, row, :) , ...
                           permute(lb.activeNode(:, :, row), [3, 1, 2, 4, 5]));
            end

            % Collect features from the tops of each tree, not the top of the feature matrix.
            topFeatures = zeros(lb.D, lb.B);
            for b = 1:lb.B
                topFeatures(:, b) = lb.features(:, b, 1, lb.N - lb.wordCounts(b) + 1, 1);
            end

            % Temporary display method.
            % lb.lattices{1}.getText()
            % permute(lb.connections(3,1,:,:), [4, 3, 1, 2])

            if hyperParams.showDetailedStats
                connectionAccuracies = correctConnectionLabels ./ totalConnectionLabels;
                connectionAccuracy = [ mean(connectionAccuracies(isfinite(connectionAccuracies))); connectionAccuracies ];
            else
                connectionAccuracy = -1;
            end
        end

        function [ wordGradients, connectionMatrixGradients, scoringVectorGradients, ...
                   compositionMatrixGradients, embeddingTransformMatrixGradients ] = ...
            getGradient(lb, incomingDeltas, wordFeatures, embeddingTransformMatrix, connectionMatrix, scoringVector, compositionMatrix, hyperParams)
            % Run backwards.

            contextPositions = -hyperParams.latticeConnectionContextWidth + 1:hyperParams.latticeConnectionContextWidth;
            connectionMatrixGradients = zeros(size(connectionMatrix, 1), size(connectionMatrix, 2), size(connectionMatrix, 3));
            scoringVectorGradients = zeros(size(scoringVector, 1), size(scoringVector, 2));
            compositionMatrixGradients = zeros(size(compositionMatrix, 1), size(compositionMatrix, 2));
            % Initialize delta matrix with the incoming deltas in the right places.

            % TODO: This could be represented as a vector covering only the deltas at one row,
            % but this could impose some time/complexity costs. Investigate.
            deltas = zeros(lb.D, lb.B, lb.N, lb.N, hyperParams.lstm + 1);
            leftEdgeEmbeddingDeltas = zeros(lb.D, 1);
            rightEdgeEmbeddingDeltas = zeros(lb.D, 1);

            % Populate the delta matrix with the incoming deltas (reasonably fast).
            for b = 1:lb.B
                deltas(:, b, 1, lb.N - lb.wordCounts(b) + 1, 1) = incomingDeltas(:, b);
            end

            % Iterate over the structure in reverse
            for row = 1:lb.N - 1
                % Get the score deltas from the softmax.
               
                % Push these deltas into the softmax that predicts where to merge.
                deltasToMerges = zeros(row, lb.B);

                % Remove deltas for inactive nodes.
                deltas(:, :, :, row, :) = ...
                    bsxfun(@times, deltas(:, :, :, row, :) , ...
                           permute(lb.activeNode(:, :, row), [3, 1, 2, 4, 5]));

                for col = 1:row
                    %% Handle composition function gradients %%

                    % Multiply in the three connection weights by the three inputs to the current features, 
                    % and add these to the existing deltas, which can either come from the inbound deltas (from the top-level classifier)
                    % or from a wide-window connection classifier.
                    deltas(:, :, col, row + 1, :) = ...
                        deltas(:, :, col, row + 1, :) + ...
                        bsxfun(@times, deltas(:, :, col, row, :), ...
                               lb.connections(1, :, col, row));
                    deltas(:, :, col + 1, row + 1, :) = ...
                        deltas(:, :, col + 1, row + 1, :) + ...
                        bsxfun(@times, deltas(:, :, col, row, :), ...
                               lb.connections(2, :, col, row));
                    compositionDeltas = ...
                        bsxfun(@times, deltas(:, :, col, row, :), ...
                               lb.connections(3, :, col, row));                 

                    % Backprop through the composition function.
                    if hyperParams.lstm
                        [ localCompositionMatrixGradients, delta_h_l, delta_h_r, delta_c_l, delta_c_r ] = ...
                            ComputeTreeLSTMLayerGradients(compositionMatrix, lb.LSTM_IFOGf(:, :, col, row), ...
                                lb.features(:, :, col, row + 1, 1), lb.features(:, :, col + 1, row + 1, 1), ...
                                lb.features(:, :, col, row + 1, 2), lb.features(:, :, col + 1, row + 1, 2), ...
                                lb.features(:, :, col, row, 2), ...
                                compositionDeltas(:, :, 1), compositionDeltas(:, :, 2));
                        deltas(:, :, col, row + 1, 1) = ...
                            deltas(:, :, col, row + 1, 1) + delta_h_l;
                        deltas(:, :, col + 1, row + 1, 1) = ...
                            deltas(:, :, col + 1, row + 1, 1) + delta_h_r;
                        deltas(:, :, col, row + 1, 2) = ...
                            deltas(:, :, col, row + 1, 2) + delta_c_l;
                        deltas(:, :, col + 1, row + 1, 2) = ...
                            deltas(:, :, col + 1, row + 1, 2) + delta_c_r;
                    else    
                        [ localCompositionMatrixGradients, compositionDeltaLeft, compositionDeltaRight ] = ...
                            ComputeRNNLayerGradients(lb.features(:, :, col, row + 1, 1), ...
                                                     lb.features(:, :, col + 1, row + 1, 1), ...
                                                     compositionMatrix, compositionDeltas, @TanhDeriv, ...
                                                     lb.compositionActivations(:, :, col, row, 1));
                        deltas(:, :, col, row + 1, 1) = ...
                            deltas(:, :, col, row + 1, 1) + compositionDeltaLeft;
                        deltas(:, :, col + 1, row + 1, 1) = ...
                            deltas(:, :, col + 1, row + 1, 1) + compositionDeltaRight;
                    end

                    % Add the composition gradients and deltas into the accumulators.
                    compositionMatrixGradients = compositionMatrixGradients + localCompositionMatrixGradients;


                    %% Handle connection function gradients %%

                    if row > 1
                        % Multiply the deltas by the three inputs to the current features to compute deltas for the connections.
                        deltasToMerges(col + 1:row, :) = bsxfun(@plus, deltasToMerges(col + 1:row, :), ...
                                                          sum(sum(deltas(:, :, col, row, :) .* ...
                                                              lb.features(:, :, col, row + 1, :), 5), 1));

                        deltasToMerges(1:col - 1, :) =  bsxfun(@plus, deltasToMerges(1:col - 1, :), ...
                                                        sum(sum(deltas(:, :, col, row, :) .* ...
                                                            lb.features(:, :, col + 1, row + 1, :), 5), 1));

                        deltasToMerges(col, :) = bsxfun(@plus, deltasToMerges(col, :), ... 
                                                    sum(sum(deltas(:, :, col, row, :) .* ...
                                                     lb.compositionActivations(:, :, col, row, :), 5), 1));
                    end
                end

                if row > 1
                    merges = permute(lb.connections(3, :, 1:row, row), [3, 2, 1, 4]);

                    % Compute gradients for the scores wrt. the incoming deltas from above and to the right.
                    [ ~, incomingDeltasToScores ] = ...
                        ComputeBareSoftmaxGradients([], merges, deltasToMerges, lb.scores(1:row, :, row));

                    % Compute gradients for the scores wrt. the independent connection supervision signal.
                    [ ~, labelDeltasToScores ] = ...
                            ComputeSoftmaxClassificationGradients([], merges, lb.connectionLabels(:, row), ...
                                lb.scores(1:row, :, row), hyperParams, hyperParams.connectionCostScale .* lb.supervisionWeights(:, row) ./ (lb.wordCounts - 2)');

                    deltasToScores = labelDeltasToScores + incomingDeltasToScores;

                    % Overwrite 0/0 deltas from inactive nodes.
                    deltasToScores(isnan(deltasToScores)) = 0;

                    % Precompute a padded version of the features.
                    paddedRow = permute(padarray(lb.features(:, :, :, row + 1, 1), [0, 0, hyperParams.latticeConnectionContextWidth - 1]), [1, 3, 2, 4]);

                    % Apply <s>/</s>
                    paddedRow(:, hyperParams.latticeConnectionContextWidth - 1, :) = ...
                        repmat(lb.leftEdgeEmbedding, [1, 1, lb.B]);
                    paddedRow(:, sub2ind([lb.N + (2 * (hyperParams.latticeConnectionContextWidth - 1)), lb.B], ...
                        max(1, hyperParams.latticeConnectionContextWidth + lb.wordCounts + row + 1 - lb.N), 1:lb.B)) = ...
                        repmat(lb.rightEdgeEmbedding, [1, 1, lb.B]);

                    for col = 1:row
                        scorerHiddenInputs = [ones(3, lb.B); ...
                            reshape(paddedRow(:, contextPositions + col + hyperParams.latticeConnectionContextWidth - 1, :), ...
                            [2 * lb.D * hyperParams.latticeConnectionContextWidth, lb.B])];
                        scorerHiddenInputs(2, :) = col;
                        scorerHiddenInputs(3, :) = (1.0 .* col) / row;
                        
                        % Dimensions: hiddenD, B
                        scorerInput = [ones(1, lb.B); lb.scorerHiddenLayer(:, :, col, row)];

                        scorerHiddenDeltas = scoringVector' * deltasToScores(col, :);
                        scorerHiddenDeltas = scorerHiddenDeltas(2:end, :) .* TanhDeriv([], lb.scorerHiddenLayer(:, :, col, row));

                        scoringVectorGradients = scoringVectorGradients + deltasToScores(col, :) * scorerInput';

                        deltasToContext = permute(reshape(connectionMatrix(:, 4:end)' * scorerHiddenDeltas, [lb.D, 2 * hyperParams.latticeConnectionContextWidth, lb.B]), [1, 3, 2]); 

                        connectionMatrixGradients = connectionMatrixGradients + ...
                                scorerHiddenDeltas * scorerHiddenInputs';

                        % Distribute the deltas from the softmax function back into its inputs.
                        sourcePositions = contextPositions(1:2 * hyperParams.latticeConnectionContextWidth) + col;
                        positions = [1:2 * hyperParams.latticeConnectionContextWidth; ...
                                     sourcePositions];
                        leftEdgePosition = find(positions(2, :) == 0);
                        positions = positions(:, positions(2, :) > 0 & positions(2, :) <= row + 1);
                        deltas(:, :, positions(2, :), row + 1, 1) = deltas(:, :, positions(2, :), row + 1, 1) + ...
                            deltasToContext(:, :, positions(1, :));

                        if ~isempty(leftEdgePosition)
                            % Handle the left padding embedding in the 0th column.
                            leftEdgeEmbeddingDeltas = leftEdgeEmbeddingDeltas ...
                                + sum(deltasToContext(:, :, leftEdgePosition), 2);
                        end

                        % Handle the right padding embedding, which can appear in any column.
                        % Surprisingly, this mess of arithmetic seems to work:
                        rightEdgePos = lb.wordCounts + row - lb.N - col + hyperParams.latticeConnectionContextWidth + 2;
                        rightEdgeIndices = [1:lb.B; rightEdgePos];
                        rightEdgeIndices = rightEdgeIndices(:, (rightEdgeIndices(2, :) > 0) ...
                            & (rightEdgeIndices(2, :) <= (2 * hyperParams.latticeConnectionContextWidth)));
                        rightEdgeEmbeddingDeltas = rightEdgeEmbeddingDeltas + sum(deltasToContext(:, ...
                            sub2ind([lb.B, 2 * hyperParams.latticeConnectionContextWidth], ...
                                    rightEdgeIndices(1, :), rightEdgeIndices(2, :))), 2);
                    end
                end

                % Scale down the deltas.
                if hyperParams.maxDeltaNorm > 0
                    multipliers = min(bsxfun(@rdivide, hyperParams.maxDeltaNorm, sum(deltas(:, :, :, row + 1, :).^2)), ones(1, lb.B, lb.N, 1, hyperParams.lstm + 1));
                    deltas(:, :, :, row + 1, :) = bsxfun(@times, deltas(:, :, :, row + 1, :), multipliers);
                end
            end

            % Run the embedding transform layers backwards.
            if hyperParams.embeddingTransformDepth > 0
                embeddingTransformMatrixGradients = zeros(size(embeddingTransformMatrix, 1), size(embeddingTransformMatrix, 2));                    
                rawEmbeddingDeltas = zeros(hyperParams.embeddingDim, lb.B, lb.N);
                
                % Remove deltas for inactive nodes.
                deltas(:, :, :, lb.N, :) = ...
                    bsxfun(@times, deltas(:, :, :, lb.N, :) , ...
                           permute(lb.activeNode(:, :, lb.N), [3, 1, 2, 4, 5]));

                for col = 1:lb.N
                    transformDeltas = deltas(:, :, col, lb.N, 1) .* lb.masks(:, :, col); % Take dropout into account
                    [ localEmbeddingTransformMatrixGradients, rawEmbeddingDeltas(:, :, col) ] = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                              transformDeltas, lb.rawEmbeddings(:, :, col), ...
                              lb.features(:, :, col, lb.N, 1), @TanhDeriv);
                    embeddingTransformMatrixGradients = embeddingTransformMatrixGradients + localEmbeddingTransformMatrixGradients;
                end

                % Handle padding embeddings.
                transformDeltas = leftEdgeEmbeddingDeltas .* lb.masks(:, 1, lb.N + 1);  % Take dropout into account
                [ localEmbeddingTransformMatrixGradients, leftEdgeEmbeddingDeltas ] = ...
                      ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                          transformDeltas, lb.rawLeftEdgeEmbedding, ...
                          lb.leftEdgeEmbedding, @TanhDeriv);
                embeddingTransformMatrixGradients = embeddingTransformMatrixGradients + localEmbeddingTransformMatrixGradients;

                transformDeltas = rightEdgeEmbeddingDeltas .* lb.masks(:, 1, lb.N + 2);  % Take dropout into account
                [ localEmbeddingTransformMatrixGradients, rightEdgeEmbeddingDeltas ] = ...
                      ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                          transformDeltas, lb.rawRightEdgeEmbedding, ...
                          lb.rightEdgeEmbedding, @TanhDeriv);
                embeddingTransformMatrixGradients = embeddingTransformMatrixGradients + localEmbeddingTransformMatrixGradients;
            else
                embeddingTransformMatrixGradients = [];
            end

            % Push deltas from the bottom row into the word gradients.
            wordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), lb.N * lb.B);

            % These have already been pooled across examples, so add them in directly.
            wordGradients(:, hyperParams.sentenceStartWordIndex) = leftEdgeEmbeddingDeltas;
            wordGradients(:, hyperParams.sentenceEndWordIndex) = rightEdgeEmbeddingDeltas;

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
                                deltas(:, b, col, lb.N, 1);
                        end
                    end
                end
            end
        end
    end
end
