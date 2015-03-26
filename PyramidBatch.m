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
        features = [];
        wordCounts = [];
        compositionInnerActivations = [];
        % connectionInnerActivations = [];  % Maybe keep later? TODO.
        connections = [];  % The above after softmax
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

            pb.features = zeros(pb.N, pb.N * pb.D, pb.B);
            pb.wordCounts = zeros(pb.B, 1);
            pb.compositionInnerActivations = zeros(pb.N - 1, (pb.N - 1) * pb.D, pb.B);
            % pb.connectionInnerActivations = zeros(pb.N - 1, pb.N - 1, pb.NUMACTIONS, pb.B);
            pb.connections = zeros(pb.N - 1, pb.N - 1, pb.NUMACTIONS, pb.B);  % The above after softmax
            pb.connectionLabels = zeros(pb.N - 1, pb.N - 1, pb.B);

            for b = 1:pb.B
                pb.wordCounts(b) = pyramids(b).wordCount;
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
        
        function [ topFeatures ] = runForward(pb, hyperParams)
            contextPositions = -hyperParams.pyramidConnectionContextWidth + 1:hyperParams.pyramidConnectionContextWidth;

            for row = pb.N - 1:-1:1
                for col = 1:row
                    % Throwaway variable, storing would take up loads of space.
                    connectionClassifierInputs = zeros((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + pb.NUMACTIONS, pb.B);
                    % TODO: Vectorize a bit more?
                    for pos = 1:2 * hyperParams.pyramidConnectionContextWidth
                        sourcePos = contextPositions(pos) + col;
                        if (sourcePos > 0) && (sourcePos <= row)
                            % I'll be shocked if there isn't an off-by-one somewhere in here.
                            connectionClassifierInputs((pos - 1) * pb.D + 1:pos * pb.D, :) = ...
                                pb.features(row + 1, (sourcePos - 1) * pb.D + 1:sourcePos * pb.D, :);
                        end
                        % Else: Leave in the zeroes. Maybe fix replace with edge-of-sentence token?
                        if col > 1
                            connectionClassifierInputs((2 * hyperParams.pyramidConnectionContextWidth) * pb.D + 1:end, :) = ...
                                pb.connections(row, col - 1, :, :);  % TODO: Need reshape?
                        end
                    end

                    connectionMatrix = rand(pb.NUMACTIONS, (2 * hyperParams.pyramidConnectionContextWidth) * pb.D + pb.NUMACTIONS + 1); % TEMP
                    pb.connections(row, col, :, :) = ComputeSoftmaxProbabilities(connectionClassifierInputs, connectionMatrix, 1:pb.NUMACTIONS); % Reshape?

                    compositionMatrix = rand(pb.D, (2 * pb.D) + 1); % TEMP
                    compositionInputs = [ones(1, pb.B); permute(pb.features(row + 1, (col - 1) * pb.D + 1:(col + 1) * pb.D, :), [2, 3, 1])];
                    pb.compositionInnerActivations(row, (col - 1) * pb.D + 1:col * pb.D, :) = compositionMatrix * compositionInputs;

                    % Apply all three connection types
                    pb.features(row, (col - 1) * pb.D + 1:col * pb.D, :) = ...
                        bsxfun(@times, tanh(pb.compositionInnerActivations(row, (col - 1) * pb.D + 1:col * pb.D, :)), ...
                                       permute(pb.connections(row, col, 3, :), [1, 2, 4, 3])) + ...
                        bsxfun(@times, pb.features(row + 1, (col - 1) * pb.D + 1:col * pb.D, :), ...
                                       permute(pb.connections(row, col, 1, :), [1, 2, 4, 3])) + ...
                        bsxfun(@times, pb.features(row + 1, col * pb.D + 1:(col + 1) * pb.D, :), ...
                                       permute(pb.connections(row, col, 2, :), [1, 2, 4, 3]));
                end

            end

            % Collect features from the tops of each tree, not the top of the feature matrix.
            topFeatures = zeros(pb.D, pb.B);
            for b = 1:pb.B
                topFeatures(:, b) = pb.features(pb.N - pb.wordCounts(b) + 1, 1:pb.D, b);
            end
        end

        % TODO: Gradients!
    end
end