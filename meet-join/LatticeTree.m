% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef LatticeTree < handle
    
    % Represents a single binary branching syntactic tree with three 
    % representations at each node:
    % - The index with which the feature vector can be looked up - if leaf
    % - The text with which the tree can be displayed.
    % - The features at the node.
    
    properties (Hidden) %TODO: Make all private.
        daughters = []; % 2 x 1 vector of trees
        text = 'NULL';
        features = []; % DIM x 1 vector
        wordIndex = -1; % -1 => Not a lexical item node.
        type = 0; % 0 - primitive symbol
                  % 1 - compose daughters using ^
                  % 2 - compose daughters using u
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
            % ( a v ( ( b u c ) v d ) )
            % (
            % cache a
            % cache v
            % (
            % (
            % cache a v b
            % cache a v b u
            % cache a v b u c
            % )
            % cache a v buc
            % cache a bc d
            % )
            % cache a bcd
            % )
            % cache abcd
            
            % if number and next is symbol:
            %   make subtree and push
            
            C = textscan(iText, '%s', 'delimiter', ' ');
            C = C{1};
             
            stack = cell(length(C));
            stackTop = 0;
            
            for i = 1:length(C)
                if ~strcmp(C{i}, '(') && ~strcmp(C{i}, ')') && ...
                        ~strcmp(C{i}, '^') && ~strcmp(C{i}, 'u')
                    % Turn words into leaf nodes
                    stack{stackTop + 1} = LatticeTree.makeLeaf(C{i}, wordMap);
                    stackTop = stackTop + 1;
                elseif strcmp(C{i}, ')')
                    % Merge at the ends of constituents
                    r = stack{stackTop};
                    c = stack{stackTop - 1};
                    l = stack{stackTop - 2};
                    stack{stackTop - 2} = LatticeTree.mergeTrees(l, c, r);
                    stackTop = stackTop - 2;
                elseif strcmp(C{i}, '^') || strcmp(C{i}, 'u')
                    stack{stackTop + 1} = C{i};
                    stackTop = stackTop + 1;     
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
            t = LatticeTree();
            t.text = iText;
            if wordMap.isKey(t.text)
                t.wordIndex = wordMap(t.text);
                t.type = 0;
            else
                disp(['Failed to map word ' t.text]);
                t.wordIndex = 1;
            end
        end
        
        function t = mergeTrees(l, c, r)
            t = LatticeTree();
            t.text = ['(', l.text, ' ', c, ' ', r.text, ')'];
            t.daughters = [l r];
            if strcmp(c, '^')
                t.type = 1;
            elseif strcmp(c, 'u')
                t.type = 2;
            else
                disp('Type not found')
                t.type = 100;
            end
            
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
            t = obj.text;
        end
        
        function f = getFeatures(obj)
            f = obj.features;
        end
        
        function type = getType(obj)
            type = obj.type;
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
                
                typeInd = obj.getType();
               
                if size(compMatrices, 1) ~= 0
                    obj.features = compNL(ComputeInnerTensorLayer( ...
                        lFeatures, rFeatures, compMatrices(:,:,:,typeInd),...
                        compMatrix(:,:,typeInd), compBias(:,typeInd)));
                else
                    obj.features = compNL(compMatrix(:,:,typeInd) * ...
                        [lFeatures; rFeatures] + compBias(:,typeInd));
                end
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
                    
            NUMCOMP = 2;
 
            upwardWordGradients = zeros(size(wordFeatures, 1),...
                                        size(wordFeatures, 2));            
            
            if size(compMatrices, 1) == 0
                upwardCompositionMatricesGradients = zeros(0, 0, 0, NUMCOMP);
            else
                upwardCompositionMatricesGradients = zeros(DIM, DIM, DIM, NUMCOMP);
            end
            upwardCompositionMatrixGradients = zeros(DIM, 2 * DIM, NUMCOMP);
            upwardCompositionBiasGradients = zeros(DIM, NUMCOMP);

            if (~isempty(obj.daughters))
                typeInd = obj.getType();
                
                lFeatures = obj.daughters(1).getFeatures();
                rFeatures = obj.daughters(2).getFeatures();
                
                if size(compMatrices, 1) == 0
                    [tempCompositionMatrixGradients, ...
                        tempCompositionBiasGradients, compDeltaLeft, ...
                        compDeltaRight] = ...
                      ComputeLayerGradients(lFeatures, rFeatures, ...
                          compMatrix(:,:,typeInd), ...
                          compBias(:,typeInd), delta, ...
                          compNLDeriv);
                    tempCompositionMatricesGradients = compMatrices;
                else
                    [tempCompositionMatricesGradients, ...
                        tempCompositionMatrixGradients, ...
                        tempCompositionBiasGradients, compDeltaLeft, ...
                        compDeltaRight] = ...
                      ComputeTensorLayerGradients(lFeatures, rFeatures, ...
                          compMatrices(:,:,:,typeInd), ...
                          compMatrix(:,:,typeInd), ...
                          compBias(:,typeInd), delta, ...
                          compNLDeriv);
                end


                  
                upwardCompositionMatricesGradients(:,:,:,typeInd) = ...
                    tempCompositionMatricesGradients;
                upwardCompositionMatrixGradients(:,:,typeInd) = ...
                    tempCompositionMatrixGradients;
                upwardCompositionBiasGradients(:,typeInd) = ...
                    tempCompositionBiasGradients;
                  
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