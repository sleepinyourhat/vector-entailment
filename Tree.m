% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef Tree < handle
    % Represents a single binary branching syntactic tree with three 
    % representations at each node:
    % - The index with which the feature vector can be looked up - if leaf
    % - The text with which the tree can be displayed.
    % - The features at the node.
    
    properties (Hidden)
        daughters = []; % 2 x 1 vector of trees
        text = 'NO_TEXT';
        features = []; % DIM x 1 vector
        mask = []; % Used in dropout
        wordIndex = -1; % -1 => Not a lexical item node.
        unknown = 0;
        type = 0; % 0 - predicate or predicate + neg
                  % 1 - quantifier
                  % 2 - neg
                  % 3 - quantifier phrase           
    end
    
    methods(Static)
        function t = printAllProperties(obj)
            disp(obj.text)
            disp(obj.features)
            disp(obj.wordIndex)
            disp(obj.type)
            disp('(')
            for daughterInd = 1:length(obj.daguhters)
                printAllProperties(obj.daguhters(daughterInd))
            end
            disp(')')
        end

        function t = makeTree(iText, wordMap)
            assert(~isempty(iText), 'Bad tree input text.');
            
            C = textscan(iText, '%s', 'delimiter', ' ');
            C = C{1};
            
            stack = cell(length(C));
            stackTop = 0;

            for i = 1:length(C)
                if ~strncmpi(C{i}, '(', 1) && ~strcmp(C{i}, ')')
                    % Turn words into leaf nodes
                    stack{stackTop + 1} = Tree.makeLeaf(C{i}, wordMap);
                    stackTop = stackTop + 1;
                elseif strcmp(C{i}, ')') && (i < 3 || ~strncmpi(C{i - 2}, '(', 1))  % Merge if the constituent isn't unary.
                    % Merge at the ends of constituents
                    r = stack{stackTop};
                    l = stack{stackTop - 1};
                    stack{stackTop - 1} = Tree.mergeTrees(l, r);
                    stackTop = stackTop - 1;
                end
            end
            
            t = stack{stackTop};
            stackTop = stackTop - 1;
            
            % Merge in any residual words to the left
            while stackTop > 0
                p = stack{stackTop};
                stackTop = stackTop - 1;
                
                t = Tree.mergeTrees(p, t);
            end

            assert((length(t.daughters) > 0 || t.wordIndex ~= -1), 'Bad tree!')
        end
        
        function t = makeLeaf(iText, wordMap)
            t = Tree();
            t.text = lower(iText);
            if wordMap.isKey(t.text)
                t.wordIndex = wordMap(t.text);
            elseif all(ismember(t.text, '0123456789.-'))
                disp(['Collapsing number ' t.text]);
                t.wordIndex = wordMap('<num>');      
                t.unknown = true;         
            else
                % Account for possible use of exactAlign
                nextTry = strtok(t.text,':');
                if wordMap.isKey(nextTry)
                    t.wordIndex = wordMap(nextTry);
                % Try splitting hyphenated words
                elseif findstr('-', nextTry)
                    [first, remainder] = strtok(nextTry, '-');
                    converted = [first, ' ( - ', remainder(2:end), ' ) '];
                    t = Tree.makeTree(converted, wordMap);
                else
                    if wordMap.isKey('<unk>')
                        t.wordIndex = wordMap('<unk>');
                        t.unknown = true;
                    else
                        assert(false, ['Failed to map word ' t.text]);
                    end
                end
            end
        end
        
        function t = mergeTrees(l, r)
            t = Tree();
            t.text = strcat(l.text, ' ', r.text);
            t.daughters = [l r];
            if l.type == 1
                t.type = 2;
            else
                t.type = l.type;
            end
                
        end
        
    end
    methods
        
        function resp = isLeaf(obj)
            resp = (isempty(obj.daughters));
        end
        
        function ld = getLeftDaughter(obj)
            if (~isempty(obj.daughters))
                ld = obj.daughters(1);
            else
                ld = 0;
            end
        end
        
        function rd = getRightDaughter(obj)
            if (length(obj.daughters) > 1)
                rd = obj.daughters(2);
            else
                rd = 0;
            end
        end
        
        function t = getText(obj)
            if isLeaf(obj)
                t = obj.text;
                if obj.unknown
                    t = [t '*'];
                end
            else
                t = [obj.getLeftDaughter().getText(), ' ', obj.getRightDaughter().getText()];
            end
        end
        
        function f = getFeatures(obj)
            % Returns the saved features for the tree.
            f = obj.features(:, 1);
        end
        
        function type = getType(obj)
            type = obj.type;
        end
        
        function i = getWordIndex(obj)
            i = obj.wordIndex;
        end

       function updateFeatures(obj, wordFeatures, compositionMatrices, ...
                                compositionMatrix, embeddingTransformMatrix, compNL, trainingMode, hyperParams)
            % Recomputes features using fresh parameters.

            if ~isempty(obj.daughters)
                obj.daughters(1).updateFeatures(...
                    wordFeatures, compositionMatrices, compositionMatrix, embeddingTransformMatrix, ...
                    compNL, trainingMode, hyperParams);

                obj.daughters(2).updateFeatures(...
                    wordFeatures, compositionMatrices, compositionMatrix, embeddingTransformMatrix, ...
                    compNL, trainingMode, hyperParams);
                
                lFeatures = obj.daughters(1).features;
                rFeatures = obj.daughters(2).features;
                
                if size(compositionMatrix, 3) <= 1 % if not untied
                    typeInd = 1;
                else
                    'Tying may not be supported...'
                    typeInd = obj.daughters(1).getType();
                end
               
                if hyperParams.useThirdOrderComposition  % TreeRNTN
                    obj.features = ComputeTensorLayer(...
                        lFeatures, rFeatures, compositionMatrices(:,:,:,typeInd),...
                        compositionMatrix(:,:,typeInd), compNL);
                elseif hyperParams.lstm  % TreeLSTM
                    [ features, cFeatures ] = ...
                            ComputeTreeLSTMLayer(compositionMatrix, lFeatures(:,1), rFeatures(:,1), ...
                                                 lFeatures(:,2), rFeatures(:,2));
                    obj.features = [features, cFeatures];
                elseif ~hyperParams.useSumming  % TreeRNN
                    obj.features = ComputeRNNLayer(lFeatures, rFeatures,...
                        compositionMatrix(:,:,typeInd), compNL);
                else  % SumNN
                    obj.features = ComputeSummingLayer(lFeatures, rFeatures);
                end
            else
                if length(embeddingTransformMatrix) == 0
                    % We have no transform layer, so just use the word features.
                    obj.features = wordFeatures(:, obj.wordIndex); 
                else
                    % Run the transfrom layer.
                    transformInnerActivations = embeddingTransformMatrix ...
                                                    * [1; wordFeatures(:, obj.wordIndex)];
                    transformActivations = compNL(transformInnerActivations);
                    [ obj.features, obj.mask ] = Dropout(transformActivations, hyperParams.bottomDropout, trainingMode, hyperParams.gpu);
                end

                if hyperParams.lstm
                    % Create blank c features
                    obj.features = [ obj.features, zeros(size(obj.features, 1), 1) ];
                end
            end
        end
        
        function [ upwardWordGradients, ...
                   upwardCompositionMatricesGradients, ...
                   upwardCompositionMatrixGradients, ...
                   upwardEmbeddingTransformMatrixGradients ] = ...
            getGradient(obj, delta, ~, wordFeatures, compositionMatrices, ...
                        compositionMatrix, embeddingTransformMatrix, ...
                        compNLDeriv, hyperParams)
            
            DIM = length(delta);
            NUMTRANS = size(embeddingTransformMatrix, 3) .* (length(embeddingTransformMatrix) > 0);

            if size(compositionMatrix, 3) == 1 % Using tied composition parameters
                NUMCOMP = 1;
            elseif size(compositionMatrix, 3) == 0 % Using summing
                NUMCOMP = 0;
            else % Untied RNN
                NUMCOMP = 3; % TODO: Hardcoded for now, here and elsewhere.
            end

            upwardWordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), 10);            
            
            if isempty(compositionMatrices)
                upwardCompositionMatricesGradients = [];
            else
                upwardCompositionMatricesGradients = zeros(size(compositionMatrices));
            end

            upwardCompositionMatrixGradients = zeros(size(compositionMatrix));
            upwardEmbeddingTransformMatrixGradients = zeros(size(embeddingTransformMatrix));

            if (~isempty(obj.daughters))
                if size(compositionMatrix, 3) == 1 % Check if using tied composition parameters
                    typeInd = 1;
                else
                    typeInd = obj.daughters(1).getType();
                end

                if hyperParams.lstm && (size(delta, 2) == 1)
                    % Add a black c delta
                    delta = [delta, zeros(DIM, 1)];
                end
                
                lFeatures = obj.daughters(1).features;
                rFeatures = obj.daughters(2).features;
                
                if hyperParams.useThirdOrderComposition  % TreeRNTN
                    [tempCompositionMatricesGradients, ...
                        tempCompositionMatrixGradients, ...
                        compDeltaLeft, compDeltaRight] = ...
                    ComputeTensorLayerGradients(lFeatures, rFeatures, ...
                          compositionMatrices(:,:,:,typeInd), ...
                          compositionMatrix(:,:,typeInd), delta, ...
                          compNLDeriv, obj.features);

                    upwardCompositionMatricesGradients(:,:,:,typeInd) = ...
                        tempCompositionMatricesGradients;
                    upwardCompositionMatrixGradients(:,:,typeInd) = ...
                        tempCompositionMatrixGradients;
                elseif hyperParams.lstm  % TreeLSTM
                    [ tempCompositionMatrixGradients, delta_h_l, delta_h_r, delta_c_l, delta_c_r ] = ...
                        ComputeTreeLSTMLayerGradients(compositionMatrix, [], ...
                            lFeatures(:,1), rFeatures(:,1), ...
                            lFeatures(:,2), rFeatures(:,2), ...
                            obj.features(:, 2), ...
                            delta(:, 1), delta(:, 2));
                    compDeltaLeft = [delta_h_l, delta_c_l];
                    compDeltaRight = [delta_h_r, delta_c_r];
                    upwardCompositionMatrixGradients(:,:,typeInd) = ...
                        tempCompositionMatrixGradients;
                elseif ~hyperParams.useSumming  % TreeRNN
                    [tempCompositionMatrixGradients, compDeltaLeft, ...
                        compDeltaRight] = ...
                    ComputeRNNLayerGradients(lFeatures, rFeatures, ...
                          compositionMatrix(:,:,typeInd), delta, ...
                          compNLDeriv, obj.features);

                    upwardCompositionMatrixGradients(:,:,typeInd) = ...
                        tempCompositionMatrixGradients;
                else  % SumNN
                    [compDeltaLeft, compDeltaRight] = ...
                      ComputeSummingLayerGradients(delta);                 
                end
                  
                % Take gradients from the left child
                [ incomingWordGradients, ...
                  incomingCompositionMatricesGradients, ...
                  incomingCompositionMatrixGradients, ...
                  incomingEmbeddingTransformMatrixGradients ] = ...
                  obj.daughters(1).getGradient( ...
                                compDeltaLeft, [], wordFeatures, ...
                                compositionMatrices, compositionMatrix, embeddingTransformMatrix, ...
                                compNLDeriv, hyperParams);
                if hyperParams.trainWords
                    upwardWordGradients = upwardWordGradients + ...
                                          incomingWordGradients;
                end
                upwardCompositionMatricesGradients = ...
                    upwardCompositionMatricesGradients + ...
                    incomingCompositionMatricesGradients;
                upwardCompositionMatrixGradients = ...
                    upwardCompositionMatrixGradients + ...
                    incomingCompositionMatrixGradients;
                upwardEmbeddingTransformMatrixGradients = ...
                    upwardEmbeddingTransformMatrixGradients + ...
                    incomingEmbeddingTransformMatrixGradients;


                
                % Take gradients from the right child
                [ incomingWordGradients, ...
                  incomingCompositionMatricesGradients, ...
                  incomingCompositionMatrixGradients, ...
                  incomingEmbeddingTransformMatrixGradients ] = ...
                  obj.daughters(2).getGradient( ...
                                compDeltaRight, [], wordFeatures, ...
                                compositionMatrices, compositionMatrix, embeddingTransformMatrix, ...
                                compNLDeriv, hyperParams);
                if hyperParams.trainWords
                    upwardWordGradients = upwardWordGradients + ...
                                          incomingWordGradients;
                end
                upwardCompositionMatricesGradients = ...
                    upwardCompositionMatricesGradients + ...
                    incomingCompositionMatricesGradients;
                upwardCompositionMatrixGradients = ...
                    upwardCompositionMatrixGradients + ...
                    incomingCompositionMatrixGradients;
                upwardEmbeddingTransformMatrixGradients = ...
                    upwardEmbeddingTransformMatrixGradients + ...
                    incomingEmbeddingTransformMatrixGradients;
            elseif hyperParams.trainWords
                % Compute gradients for embedding transform layers
                delta = delta(:, 1);  % Ignore c deltas if present.

                if NUMTRANS > 0
                    delta = delta .* obj.mask; % Take dropout into account
                    [ upwardEmbeddingTransformMatrixGradients, delta ] = ...
                          ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                              delta, wordFeatures(:, obj.wordIndex), ...
                              obj.features(:, 1), compNLDeriv, hyperParams.gpu);
                end

                % Compute the word feature gradients
                upwardWordGradients(:, obj.wordIndex) = ...
                    upwardWordGradients(:, obj.wordIndex) + delta;
            elseif NUMTRANS > 0
                % Compute gradients for embedding transform layers
                delta = delta(:, 1);  % Ignore c deltas if present.

                delta = delta .* obj.mask; % Take dropout into account

                upwardEmbeddingTransformMatrixGradients = ...
                      ComputeEmbeddingTransformGradients(embeddingTransformMatrix, ...
                          delta, wordFeatures(:, obj.wordIndex), ...
                          obj.features(:, 1), compNLDeriv, hyperParams.gpu);
            end            
        end
    end
end
