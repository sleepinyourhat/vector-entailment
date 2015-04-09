% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef Sequence < handle
    % Represents a single word sequence (sentence) with three 
    % representations at each node:
    % - The index with which the feature vector can be looked up - if leaf
    % - The text with which the sequence can be displayed.
    % - The features at the node.

    % Sentences can generally be represented by the handle of the final node in the sentence.
    
    % NOTE: There are two ways to train or evaluate a model using Sequence objects. One is to
    % use the runForward and getGradients functions in this object. The other is to create 
    % SequenceBatch objects from a set of Sequence objects and call the functions there.
    % The functions in this file are designed to handle sequences of arbitrary length, and to 
    % follow the same object-oriented interface as the tree model in this package. The 
    % SequenceBatch alternative is likely much faster, if harder to read and edit.


    properties
        pred = []; % the preceeding node or empty
        text = 'NO_TEXT';
        activations = []; % DIM x 1 vector
        cActivations = []; % DIM x 1 vector - LSTM use only
        inputActivations = []; % DIM x 1 vector - the word vector (after transformation if applicable)
        mask = []; % Used in dropout
        activationCache = []; % Equivalent to activationsPreNL for RNNs and IFOGf for LSTMs.
        wordIndices = []; % The fill list of word indices, stored only at the last node.
        wordCount = -1; % The total number of words, stored only at the last node.
        wordIndex = -1; % -1 => Not a lexical item node.
        unknown = 0;   
    end

    methods(Static)
        function s = makeSequence(iText, wordMap, useParens)
            assert(~isempty(iText), 'Bad input text.');
            
            C = textscan(iText, '%s', 'delimiter', ' ');
            C = C{1};
            s = [];
            wordIndices = zeros(length(C), 1);
            numWords = 0;

            for i = 1:length(C)
                if useParens || ~(strcmp(C{i}, '(') || strcmp(C{i}, ')'))
                    % Turn words into nodes
                    s = Sequence.makeNode(C{i}, s, wordMap);
                    numWords = numWords + 1;
                    wordIndices(numWords) = s.wordIndex;
                end
            end
            s.wordCount = numWords;
            s.wordIndices = wordIndices(1:numWords);
        end
        
        function s = makeNode(iText, pred, wordMap)
            s = Sequence();
            s.text = lower(iText);
            s.pred = pred;
                
            if wordMap.isKey(s.text)
                s.wordIndex = wordMap(s.text);
            elseif all(ismember(s.text, '0123456789.-'))
                disp(['Collapsing number ' s.text]);
                s.wordIndex = wordMap('<num>'); 
                s.unknown = true;              
            else
                % Account for possible use of exactAlign
                nextTry = strtok(s.text,':');
                if wordMap.isKey(nextTry)
                    s.wordIndex = wordMap(nextTry);
                % Try splitting hyphenated words
                elseif findstr('-', nextTry)
                    [first, remainder] = strtok(nextTry, '-');
                    s = Sequence.makeNode(first, pred, wordMap);
                    s = Sequence.makeNode('-', s, wordMap);
                    s = Sequence.makeNode(remainder, s, wordMap);
                else
                    if wordMap.isKey('<unk>')
                        s.wordIndex = wordMap('<unk>');
                        s.unknown = true;
                    else
                        assert(false, ['Failed to map word ' s.text]);
                    end
                end
            end
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
                if obj.unknown
                    s = [s '*'];
                end
            else
                s = [obj.getPred().getText(), ' ', obj.text()];
            end
        end

        
        function f = getFeatures(obj)
            % Returns the saved features for the node.
            f = obj.activations;
        end
                
        function f = getCFeatures(obj)
            % Returns the saved features for the node.
            f = obj.cActivations;
        end
        
        function i = getWordIndex(obj)
            i = obj.wordIndex;
        end
        
        function i = getWordList(obj)
            i = obj.wordIndex;
        end

        function clearActivations(obj)
            obj.activations = []; % DIM x 1 vector
            obj.cActivations = []; % DIM x 1 vector - LSTM use only
            obj.inputActivations = []; % DIM x 1 vector - the word vector (after transformation if applicable)
            obj.mask = []; % Used in dropout
            obj.activationCache = []; % Equivalent to activationsPreNL for RNNs and IFOGf for LSTMs.
        end

        function updateFeatures(obj, wordFeatures, compMatrices, ...
                                compMatrix, embeddingTransformMatrix, compNL, trainingMode, hyperParams)
            % Recomputes features using fresh parameters.

            LSTM = size(compMatrix, 1) > size(compMatrix, 2);

            % Compute a feature vector for the input at this node.
            if length(embeddingTransformMatrix) == 0
                % We have no transform layer, so just use the word features.
                obj.inputActivations = wordFeatures(:, obj.wordIndex);
            else
                % Run the transform layer.
                transformInnerActivations = embeddingTransformMatrix ...
                                                * [1; wordFeatures(:, obj.wordIndex)];

                transformActivations = compNL(transformInnerActivations);

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
        
        function [ forwardWordGradients, ...
                   forwardCompositionMatricesGradients, ...
                   forwardCompositionMatrixGradients, ...
                   forwardEmbeddingTransformMatrixGradients ] = ...
            getGradient(obj, deltaH, deltaC, wordFeatures, compMatrices, ...
                        compMatrix, embeddingTransformMatrix, ...
                        compNLDeriv, hyperParams)            
            LSTM = size(compMatrix, 1) > size(compMatrix, 2);
            HIDDENDIM = length(deltaH);
            EMBDIM = size(wordFeatures, 1);

            if ~isempty(embeddingTransformMatrix)
                INPUTDIM = size(embeddingTransformMatrix, 1);
            else
                INPUTDIM = EMBDIM;
            end
            NUMTRANS = size(embeddingTransformMatrix, 3);
            if isempty(embeddingTransformMatrix)
                NUMTRANS = 0;
            end

            forwardWordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), 10);            

            forwardCompositionMatricesGradients = [];            
            forwardEmbeddingTransformMatrixGradients = zeros(HIDDENDIM, EMBDIM + 1, NUMTRANS);

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

            else
               if ~isempty(obj.pred)
                    predActivations = obj.pred.activations;
                else
                    predActivations = zeros(size(compMatrix, 1), 1);
                end

                [ forwardCompositionMatrixGradients, compDeltaPred, ...
                    compDeltaInput ] = ...
                ComputeRNNLayerGradients(predActivations, obj.inputActivations, ...
                      compMatrix, deltaH, ...
                      compNLDeriv, obj.activations);
                compDeltaPredC = [];
            end
                

            if ~isempty(obj.pred)
                % Take gradients from the predecessor
                [ incomingWordGradients, incomingCompositionMatricesGradients, ...
                  incomingCompositionMatrixGradients, incomingEmbeddingTransformMatrixGradients ] = ...
                  obj.pred.getGradient( ...
                                compDeltaPred, compDeltaPredC, wordFeatures,  compMatrices, ...
                                compMatrix, embeddingTransformMatrix, ...
                                compNLDeriv, hyperParams);
                if hyperParams.trainWords
                    forwardWordGradients = forwardWordGradients + ...
                                          incomingWordGradients;
                end

                forwardCompositionMatrixGradients = ...
                    forwardCompositionMatrixGradients + ...
                    incomingCompositionMatrixGradients;
                forwardEmbeddingTransformMatrixGradients = ...
                    forwardEmbeddingTransformMatrixGradients + ...
                    incomingEmbeddingTransformMatrixGradients;
            end
            
            if hyperParams.trainWords
                % Compute gradients for embedding transform layers and words

                if NUMTRANS > 0
                    compDeltaInput = compDeltaInput .* obj.mask; % Take dropout into account
                    [tempEmbeddingTransformMatrixGradients, compDeltaInput] = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                              compDeltaInput, wordFeatures(:, obj.wordIndex), ...
                              obj.inputActivations, compNLDeriv);
                    forwardEmbeddingTransformMatrixGradients = ...
                        forwardEmbeddingTransformMatrixGradients + ...
                        tempEmbeddingTransformMatrixGradients;
                end

                % Compute the word feature gradients
                forwardWordGradients(:, obj.wordIndex) = ...
                    forwardWordGradients(:, obj.wordIndex) + compDeltaInput;
            elseif NUMTRANS > 0
                % Compute gradients for embedding transform layers only
                compDeltaInput = compDeltaInput .* obj.mask; % Take dropout into account

                [tempEmbeddingTransformMatrixGradients, ~] = ...
                      ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                          compDeltaInput, wordFeatures(:, obj.wordIndex), ...
                          obj.inputActivations, compNLDeriv);
                forwardEmbeddingTransformMatrixGradients = ...
                    forwardEmbeddingTransformMatrixGradients + ...
                    tempEmbeddingTransformMatrixGradients;
            end            
        end
    end
end
