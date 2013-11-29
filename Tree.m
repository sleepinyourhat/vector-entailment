classdef Tree < handle
    
    % Represents a single binary branching syntactic tree with three representations at each node:
    % - The index with which the feature vector can be looked up - IF LEAF
    % - The text with which the tree can be displayed.
    % - The features at the node.
    
    properties (Hidden) %TODO: Make all private.
        daughters = []; % 2 x 1 vector of trees
        text = 'NULL';
        features = []; % DIM x 1 vector
        wordIndex = -1; % -1 => Not a lexical item node.
    end
    
    methods(Static)
        
        function t = makeTree(iText, wordMap)
            % Parsing strategy:          
            % ( a b ) ( c d )
            % (
            %  cache - a
            %  cache - a b
            % )
            % cache -> ab - merge last two nodes
            % (
            %  cache ab c
            %  cache ab c d
            % )
            % cache ab cd
            % cache abcd
            % 
            % 
            % ( a ( ( b c ) d ) )
            % (
            % cache a
            % (
            % (
            % cache a b
            % cache a b c
            % )
            % cache a bc
            % cache a bc d
            % )
            % cache a bcd
            % )
            % cache abcd
            
            C = textscan(iText, '%s', 'delimiter', ' ');
            C = C{1};
             
            stack = cell(length(C));
            stackTop = 0;
            
            for i = 1:length(C)
                if ~strcmp(C{i}, '(') && ~strcmp(C{i}, ')')
                    % Turn words into leaf nodes
                    stack{stackTop + 1} = Tree.makeLeaf(C{i}, wordMap);
                    stackTop = stackTop + 1;
                elseif strcmp(C{i}, ')')
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
        end
        
        function t = makeLeaf(iText, wordMap)
            t = Tree();
            t.text = iText;
            if wordMap.isKey(t.text)
                t.wordIndex = wordMap(t.text);
            else
                disp(['Failed to map word ' t.text]);
                t.wordIndex = 1;
            end
        end
        
        function t = mergeTrees(l, r)
            t = Tree();
            t.text = strcat(l.text, ' ', r.text);
            t.daughters = [l r];
        end
        
    end
    methods
        
        function resp = isLeaf(obj)
            resp = (isempty(obj.daughters)); % TODO: Fill in for undefined.
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
            else
                t = [obj.getLeftDaughter().getText(), ' ', obj.getRightDaughter().getText()];
            end
        end
        
        function f = getFeatures(obj)
            f = obj.features;
        end
        
        function i = getWordIndex(obj)
            i = obj.wordIndex;
        end
        
        function updateFeatures(obj, wordFeatures, compMatrices, ...
                                compMatrix, compBias, compNL)
            %if nargin < 5
            %    % Use default (averaging) composition function
            %    dim = size(wordFeatures, 2);
            %    compMatrices = zeros(dim , (dim ^ 2));
            %    compMatrix = [eye(dim), eye(dim)];
            %    compBias = zeros(dim, 1);
            %end
            
            if (~isempty(obj.daughters))
                for daughterIndex = 1:length(obj.daughters)
                    obj.daughters(daughterIndex).updateFeatures(...
                        wordFeatures, compMatrices, compMatrix, compBias, ...
                        compNL);
                end
                
                lFeatures = obj.daughters(1).getFeatures();
                rFeatures = obj.daughters(2).getFeatures();
                
                obj.features = compNL(ComputeInnerTensorLayer( ...
                    lFeatures, rFeatures, compMatrices, compMatrix, compBias));
            else
                % In this case, we are a leaf.
                obj.features = wordFeatures(obj.wordIndex, :)';
            end
        end
        
        function [ upwardWordGradients, ...
                   upwardCompositionMatricesGradients, ...
                   upwardCompositionMatrixGradients, ...
                   upwardCompositionBiasGradients ] = ...
            getGradient(obj, delta, wordFeatures, compMatrices, ...
                        compMatrix, compBias, compNLDeriv)
                    % Delta should be a column vector.
                    
            DIM = size(compBias, 1);
            
            upwardWordGradients = sparse([], [], [], ...
                size(wordFeatures, 1), size(wordFeatures, 2), 10);            
            upwardCompositionMatricesGradients = zeros(DIM, DIM, DIM);
            upwardCompositionMatrixGradients = zeros(DIM, 2 * DIM);
            upwardCompositionBiasGradients = zeros(DIM, 1);

            % Is this necessary? No. We have to have done this to get the
            % objective.
            % obj.updateFeatures(wordFeatures, compMatrix, compMatrix, compBias);
            
            if (~isempty(obj.daughters))
                lFeatures = obj.daughters(1).getFeatures();
                rFeatures = obj.daughters(2).getFeatures();
               
                
                [upwardCompositionMatricesGradients, ...
                    upwardCompositionMatrixGradients, ...
                    upwardCompositionBiasGradients, compDeltaLeft, ...
                    compDeltaRight] = ...
                  ComputeTensorLayerGradients(lFeatures, rFeatures, ...
                      compMatrices, compMatrix, compBias, delta, ...
                      compNLDeriv);

                % Take gradients from below.
                [ incomingWordGradients, ...
                  incomingCompositionMatricesGradients, ...
                  incomingCompositionMatrixGradients, ...
                  incomingCompositionBiasGradients ] = ...
                  obj.getLeftDaughter.getGradient( ...
                                compDeltaLeft, wordFeatures, ...
                                compMatrices, compMatrix, compBias, ...
                                compNLDeriv);
                upwardWordGradients = upwardWordGradients + ...
                                      incomingWordGradients;
                upwardCompositionMatricesGradients = ...
                    upwardCompositionMatricesGradients + ...
                    incomingCompositionMatricesGradients;
                upwardCompositionMatrixGradients = ...
                    upwardCompositionMatrixGradients + ...
                    incomingCompositionMatrixGradients;
                upwardCompositionBiasGradients = ...
                    upwardCompositionBiasGradients + ...
                    incomingCompositionBiasGradients;
                
                % Take gradients from below.
                [ incomingWordGradients, ...
                  incomingCompositionMatricesGradients, ...
                  incomingCompositionMatrixGradients, ...
                  incomingCompositionBiasGradients ] = ...
                  obj.getRightDaughter.getGradient( ...
                                compDeltaRight, wordFeatures, ...
                                compMatrices, compMatrix, compBias, ...
                                compNLDeriv);
                upwardWordGradients = upwardWordGradients + ...
                                      incomingWordGradients;
                upwardCompositionMatricesGradients = ...
                    upwardCompositionMatricesGradients + ...
                    incomingCompositionMatricesGradients;
                upwardCompositionMatrixGradients = ...
                    upwardCompositionMatrixGradients + ...
                    incomingCompositionMatrixGradients;
                upwardCompositionBiasGradients = ...
                    upwardCompositionBiasGradients + ...
                    incomingCompositionBiasGradients;
            else 
               % Compute word feature gradients here.
               upwardWordGradients(obj.getWordIndex, :) = ...
                   upwardWordGradients(obj.getWordIndex, :) + delta';
            end                
        end
    end
end