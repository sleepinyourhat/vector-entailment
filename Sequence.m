% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef Sequence < handle
    % Represents a single word sequence (sentence) with three 
    % representations at each node:
    % - The index with which the feature vector can be looked up - if leaf
    % - The text with which the sequence can be displayed.
    % - The features at the node.
    
    properties (Hidden)
        pred = []; % the preceeding node or empty
        text = 'NO_TEXT';
        hiddenFeatures = []; % DIM x 1 vector
        inputFeatures = [];
        mask = []; % Used in dropout
        hiddenFeaturesPreNL = [];
        wordIndex = -1; % -1 => Not a lexical item node.
        transformInnerActivations = []; % Stored activations for the embedding tranform layers.       
    end
    
    methods(Static)
        function s = printAllProperties(obj)
            if ~isempty(pred)
                printAllProperties(pred)
            end

            disp(obj.text)
            disp(obj.inputFeatures)
            disp(obj.hiddenFeatures)
            disp(obj.hiddenFeaturesPreNL)
            disp(obj.wordIndex)
            disp(obj.transformInnerActivations)
        end

        function s = makeSequence(iText, wordMap)
            assert(~isempty(iText), 'Bad input text.');
            
            C = textscan(iText, '%s', 'delimiter', ' ');
            C = C{1};
            s = [];
            
            for i = 1:length(C)
                % Turn words into nodes
                s = Sequence.makeNode(C{i}, pred, wordMap);
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
            f = obj.hiddenFeatures;
        end
        
        function i = getWordIndex(obj)
            i = obj.wordIndex;
        end
        
        function updateFeatures(obj, wordFeatures, compMatrices, ...
                                compMatrix, compBias,  embeddingTransformMatrix, embeddingTransformBias, compNL, dropout)
            % Recomputes features using fresh parameters.

            % Compute a feature vector for the input at this node.
            if length(embeddingTransformMatrix) == 0
                % We have no transform layer, so just use the word features.
                obj.inputFeatures = wordFeatures(obj.wordIndex, :)'; 
            else
                % Run the transfrom layer.
                obj.transformInnerActivations = embeddingTransformMatrix ...
                                                * wordFeatures(obj.wordIndex, :)' + ...
                                                embeddingTransformBias;

                activations = compNL(obj.transformInnerActivations);

                [obj.inputFeatures, obj.mask] = Dropout(activations, dropout);
            end

            % Compute a feature vector for the predecessor node.
            if (~isempty(obj.pred))
                obj.pred.updateFeatures(...
                    wordFeatures, compMatrix, compBias, embeddingTransformMatrix, embeddingTransformBias, ...
                    compNL, dropout);
                
                predFeatures = obj.pred.features;
            else
                predFeatures = zeros(1, size(compMatrix, 1))
            end    

            % Update the hidden features.
            [obj.hiddenFeatures, obj.hiddenFeaturesPreNL] = ComputeRNNLayer(predFeatures, obj.inputFeatures,...
                compMatrix, compBias, compNL);
            end
        end
        
        function [ forwardWordGradients, ...
                   forwardCompositionMatricesGradients, ...
                   forwardCompositionMatrixGradients, ...
                   forwardCompositionBiasGradients, ...
                   forwardEmbeddingTransformMatrixGradients, ...
                   forwardEmbeddingTransformBiasGradients ] = ...
            getGradient(obj, delta, wordFeatures, compMatrices, ...
                        compMatrix, compBias, embeddingTransformMatrix, embeddingTransformBias, ...
                        compNLDeriv, hyperParams)
            % Note: Delta should be a column vector.
            
            HIDDENDIM = length(delta);
            EMBDIM = size(wordFeatures, 2);

            if ~isempty(embeddingTransformMatrix)
                INPUTDIM = size(embeddingTransformMatrix, 1);
            else
                INPUTDIM = EMBDIM;
            end
            NUMTRANS = size(embeddingTransformMatrix, 3);

            forwardWordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), 10);            
            
            forwardEmbeddingTransformMatrixGradients = zeros(DIM, EMBDIM, NUMTRANS);
            forwardEmbeddingTransformBiasGradients = zeros(DIM, NUMTRANS);

            if ~isempty(obj.pred)
                predFeatures = obj.pred.hiddenFeatures;
            else
                predFeatures = zeros(1, size(compMatrix, 1))
            end
            
            [forwardCompositionMatrixGradients, ...
                forwardCompositionBiasGradients, compDeltaPred, ...
                compDeltaInput] = ...
            ComputeRNNLayerGradients(predFeatures, inputFeatures, ...
                  compMatrix, compBias, delta, ...
                  compNLDeriv, obj.featuresPreNL);
              
            % Take gradients from the predecessor
            [ incomingWordGradients, ...
              incomingCompositionMatrixGradients, ...
              incomingCompositionBiasGradients, ...
              incomingEmbeddingTransformMatrixGradients, ...
              incomingEmbeddingTransformBiasGradients ] = ...
              obj.pred.getGradient( ...
                            compDeltaPred, wordFeatures, ...
                            compMatrix, compBias, embeddingTransformMatrix, embeddingTransformBias, ...
                            compNLDeriv, hyperParams);
            if hyperParams.trainWords
                forwardWordGradients = forwardWordGradients + ...
                                      incomingWordGradients;
            end
            forwardCompositionMatricesGradients = ...
                forwardCompositionMatricesGradients + ...
                incomingCompositionMatricesGradients;
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
            
            if hyperParams.trainWords
                % Compute gradients for embedding transform layers and words

                if NUMTRANS > 0
                    delta = delta .* obj.mask; % Take dropout into account
                    [tempEmbeddingTransformMatrixGradients, ...
                          tempEmbeddingTransformBiasGradients, delta] = ...
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
                    forwardWordGradients(obj.getWordIndex, :) + delta';
            elseif NUMTRANS > 0
                % Compute gradients for embedding transform layers only
                delta = delta .* obj.mask; % Take dropout into account

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