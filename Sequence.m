% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef Sequence < handle
    % Represents a single word sequence (sentence) with three 
    % representations at each node:
    % - The index with which the feature vector can be looked up - if leaf
    % - The text with which the sequence can be displayed.
    % - The features at the node.
    
    % Sentences can generally be represented by the handle of the final node in the sentence.

    properties (Hidden)
        pred = []; % the preceeding node or empty
        text = 'NO_TEXT';
        activations = []; % DIM x 1 vector
        cActivations = []; % DIM x 1 vector - LSTM use only
        inputActivations = []; % DIM x 1 vector - the word vector (after transformation if applicable)
        mask = []; % Used in dropout
        activationCache = []; % Equivalent to activationsPreNL for RNNs and IFOGf for LSTMs.
        wordIndex = -1; % -1 => Not a lexical item node.
        transformInnerActivations = []; % Stored activations for the embedding tranform layers.       
    end
    
    methods(Static)
        function s = makeSequence(iText, wordMap, useParens)
            assert(~isempty(iText), 'Bad input text.');
            
            C = textscan(iText, '%s', 'delimiter', ' ');
            C = C{1};
            s = [];
            
            for i = 1:length(C)
                if useParens || ~(strcmp(C{i}, '(') || strcmp(C{i}, ')'))
                    % Turn words into nodes
                    s = Sequence.makeNode(C{i}, s, wordMap);
                end
            end
        end
        
        function s = makeNode(iText, pred, wordMap)
            s = Sequence();
            s.text = lower(iText);
            s.pred = pred;
            if wordMap.isKey(s.text)
                s.wordIndex = wordMap(s.text);
            elseif all(ismember(s.text, '0123456789.-'))
                disp(['Collapsing number ' s.text]);
                s.wordIndex = wordMap('*NUM*');               
            else
                nextTry = strtok(s.text,':');
                if wordMap.isKey(nextTry)
                    s.wordIndex = wordMap(nextTry);
                else
                    if rand > 0.99 % Downsample what gets logged.
                        disp(['Failed to map word ' s.text]);
                    end
                    s.wordIndex = wordMap('*UNK*');
                end
            end
            assert(s.wordIndex ~= -1, 'Bad leaf!')
        end
    end

    methods
        function printAllProperties(obj)
            if ~isempty(obj.pred)
                printAllProperties(obj.pred)
            end

            disp(obj.text)
            disp(obj.inputActivations)
            disp(obj.activations)
            disp(obj.activationCache)
            disp(obj.wordIndex)
            disp(obj.transformInnerActivations)
        end

        function resp = isStart(obj)
            resp = (isempty(obj.pred));
        end
        
        function p = getPred(obj)
            p = obj.pred;
        end
        
        function s = getText(obj)
            if isStart(obj)
                s = obj.text;
            else
                s = [obj.getPred().getText(), ' ', obj.text()];
            end
        end
        
        function f = getFeatures(obj)
            % Returns the saved features for the node.
            f = obj.activations;
        end
        
        function i = getWordIndex(obj)
            i = obj.wordIndex;
        end
        
        function updateFeatures(obj, wordFeatures, compMatrices, ...
                                compMatrix, compBias, embeddingTransformMatrix, embeddingTransformBias, compNL, dropout)
            % Recomputes features using fresh parameters.

            LSTM = isempty(compBias);

            % Compute a feature vector for the input at this node.
            if length(embeddingTransformMatrix) == 0
                % We have no transform layer, so just use the word features.
                obj.inputActivations = wordFeatures(obj.wordIndex, :)';
            else
                % Run the transform layer.
                obj.transformInnerActivations = embeddingTransformMatrix ...
                                                * wordFeatures(obj.wordIndex, :)' + ...
                                                embeddingTransformBias;

                activations = compNL(obj.transformInnerActivations);

                [obj.inputActivations, obj.mask] = Dropout(activations, dropout);
            end

            % Compute a feature vector for the predecessor node.
            if ~isempty(obj.pred)
                obj.pred.updateFeatures(...
                    wordFeatures, compMatrices, compMatrix, compBias, embeddingTransformMatrix, embeddingTransformBias, ...
                    compNL, dropout);
                
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
                [ obj.activations, obj.activationCache ] = ComputeRNNLayer(predActivations, obj.inputActivations,...
                    compMatrix, compBias, compNL);
            end
        end
        
        function [ forwardWordGradients, ...
                   forwardCompositionMatricesGradients, ...
                   forwardCompositionMatrixGradients, ...
                   forwardCompositionBiasGradients, ...
                   forwardEmbeddingTransformMatrixGradients, ...
                   forwardEmbeddingTransformBiasGradients ] = ...
            getGradient(obj, deltaH, deltaC, wordFeatures, compMatrices, ...
                        compMatrix, compBias, embeddingTransformMatrix, embeddingTransformBias, ...
                        compNLDeriv, hyperParams)
            % Note: Delta should be a column vector.
            
            LSTM = isempty(compBias);
            HIDDENDIM = length(deltaH);
            EMBDIM = size(wordFeatures, 2);

            if ~isempty(embeddingTransformMatrix)
                INPUTDIM = size(embeddingTransformMatrix, 1);
            else
                INPUTDIM = EMBDIM;
            end
            NUMTRANS = size(embeddingTransformMatrix, 3);

            forwardWordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), 10);            

            forwardCompositionMatricesGradients = [];            
            forwardEmbeddingTransformMatrixGradients = zeros(HIDDENDIM, EMBDIM, NUMTRANS);
            forwardEmbeddingTransformBiasGradients = zeros(HIDDENDIM, NUMTRANS);

            if LSTM
                if ~isempty(obj.pred)
                    predC = obj.pred.cActivations;
                    predH = obj.pred.activations;
                else
                    predC = zeros(size(obj.cActivations, 1), 1);
                    predH = zeros(size(obj.activations, 1), 1);
                end

                if isempty(deltaC)
                    deltaC = 0 .* deltaH;
                end

                if ~isempty(obj.pred)
                    [ forwardCompositionMatrixGradients, compDeltaInput, compDeltaPred, compDeltaPredC ] ...
                        = ComputeLSTMLayerGradients(obj.inputActivations, compMatrix, obj.activationCache, ...
                            predC, predH, obj.cActivations, deltaH, deltaC);
                else
                    [ forwardCompositionMatrixGradients, compDeltaInput ] ...
                        = ComputeLSTMLayerGradients(obj.inputActivations, compMatrix, obj.activationCache, ...
                            predC, predH, obj.cActivations, deltaH, deltaC);                 
                end
                    

                forwardCompositionBiasGradients = [];
            else
               if ~isempty(obj.pred)
                    predActivations = obj.pred.activations;
                else
                    predActivations = zeros(size(compMatrix, 1), 1);
                end

                [ forwardCompositionMatrixGradients, ...
                    forwardCompositionBiasGradients, compDeltaPred, ...
                    compDeltaInput ] = ...
                ComputeRNNLayerGradients(predActivations, obj.inputActivations, ...
                      compMatrix, compBias, deltaH, ...
                      compNLDeriv, obj.activationCache);
                compDeltaPredC = [];
            end
                

            if ~isempty(obj.pred)
                % Take gradients from the predecessor
                [ incomingWordGradients, incomingCompositionMatricesGradients, ...
                  incomingCompositionMatrixGradients, incomingCompositionBiasGradients, ...
                  incomingEmbeddingTransformMatrixGradients, ...
                  incomingEmbeddingTransformBiasGradients ] = ...
                  obj.pred.getGradient( ...
                                compDeltaPred, compDeltaPredC, wordFeatures,  compMatrices, ...
                                compMatrix, compBias, embeddingTransformMatrix, embeddingTransformBias, ...
                                compNLDeriv, hyperParams);
                if hyperParams.trainWords
                    forwardWordGradients = forwardWordGradients + ...
                                          incomingWordGradients;
                end

                forwardCompositionMatrixGradients = ...
                    forwardCompositionMatrixGradients + ...
                    incomingCompositionMatrixGradients;
                forwardCompositionBiasGradients = ...
                    forwardCompositionBiasGradients + ...
                    incomingCompositionBiasGradients;
                forwardEmbeddingTransformMatrixGradients = ...
                    forwardEmbeddingTransformMatrixGradients + ...
                    incomingEmbeddingTransformMatrixGradients;
                forwardEmbeddingTransformBiasGradients = ...
                    forwardEmbeddingTransformBiasGradients + ...
                    incomingEmbeddingTransformBiasGradients;
            end
            
            if hyperParams.trainWords
                % Compute gradients for embedding transform layers and words

                if NUMTRANS > 0
                    compDeltaInput = compDeltaInput .* obj.mask; % Take dropout into account
                    [tempEmbeddingTransformMatrixGradients, ...
                          tempEmbeddingTransformBiasGradients, compDeltaInput] = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, embeddingTransformBias, ...
                              compDeltaInput, wordFeatures(obj.wordIndex, :)', ...
                              obj.transformInnerActivations, compNLDeriv);
                    forwardEmbeddingTransformMatrixGradients = ...
                        forwardEmbeddingTransformMatrixGradients + ...
                        tempEmbeddingTransformMatrixGradients;
                    forwardEmbeddingTransformBiasGradients = ...
                        forwardEmbeddingTransformBiasGradients + ...
                        tempEmbeddingTransformBiasGradients;
                end

                % Compute the word feature gradients
                forwardWordGradients(obj.getWordIndex, :) = ...
                    forwardWordGradients(obj.getWordIndex, :) + compDeltaInput';
            elseif NUMTRANS > 0
                % Compute gradients for embedding transform layers only
                compDeltaInput = compDeltaInput .* obj.mask; % Take dropout into account

                [tempEmbeddingTransformMatrixGradients, ...
                      tempEmbeddingTransformBiasGradients, ~] = ...
                      ComputeEmbeddingTransformGradients(embeddingTransformMatrix, embeddingTransformBias, ...
                          compDeltaInput, wordFeatures(obj.wordIndex, :)', ...
                          obj.transformInnerActivations, compNLDeriv);
                forwardEmbeddingTransformMatrixGradients = ...
                    forwardEmbeddingTransformMatrixGradients + ...
                    tempEmbeddingTransformMatrixGradients;
                forwardEmbeddingTransformBiasGradients = ...
                    forwardEmbeddingTransformBiasGradients + ...
                    tempEmbeddingTransformBiasGradients;
            end            
        end
    end
end