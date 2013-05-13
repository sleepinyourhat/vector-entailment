classdef Tree
    
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
        % TODO: Constructors for multiword trees?
        
        function t = makeLeaf(iText)
            global wordMap;
            global wordFeatures;
            t = Tree();
            t.text = iText;
            t.wordIndex = wordMap(t.text);
            t.features = wordFeatures(t.wordIndex, :);
        end
        
    end
    methods
        
        function resp = isLeaf(obj)
            resp = (length(obj.daughters) == 0); % TODO: Fill in for undefined.
        end
        
        function ld = getLeftDaughter(obj)
            if (length(obj.daughters) > 0)
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
        
        function updateFeatures(obj)
            global wordFeatures;
            
            if (~isempty(obj.daughters))
                for (daughterIndex = 1:length(obj.daughters))
                    obj.daughters(daughterIndex).updateFeatures();
                end
                % TODO: APPLY COMPOSITION FUNCTION.
            else
                % In this case, we are a leaf.
                features = wordFeatures(:, obj.wordIndex);
            end
        end
    end
end