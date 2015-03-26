% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef Pyramid < handle
    % Represents a single binary branching syntactic tree with three 
    % representations at each node:
    % - The index with which the feature vector can be looked up - if leaf
    % - The text with which the tree can be displayed.
    % - The features at the node.
    
    properties
        wordIndices = []; % N ints
        wordCount = -1; % int
        connectionLabels = [] % optional (N - 1) x (N - 1) matrix
        text = 'NO TEXT'; % string
        % TODO: Dropout
        % transformInnerActivations = []; % Stored activations for the embedding tranform layers. TODO.      
    end
    
    methods(Static)
        function p = makePyramid(iText, wordMap)
            p = Pyramid();

            terms = textscan(iText, '%s', 'delimiter', ' ');

            assert(strcmp(terms{1}{1}, '(') && strcmp(terms{1}{end}, ')'), 'Strings must be parsed, including outermost parens.');

            p.wordCount = (length(terms{1}) + 2) / 3;  % Generalizes for binary parse trees.
            p.wordIndices = zeros(p.wordCount, 1);
            p.connectionLabels = zeros(p.wordCount - 1, p.wordCount - 1);
            p.text = iText; % B-dim cell of strings

            depth = p.wordCount;
            mergeCount = 0;
            wordIndex = 0;

            % Load the words and the tree structure
            for t = 1:length(terms{1})
                if strcmp(terms{1}{t}, ')')  % Mark the merge in the tree structure.
                    p.connectionLabels(depth - 1, wordIndex - 1 - mergeCount) = 3;
                    depth = depth - 1;
                    mergeCount = mergeCount + 1;
                elseif ~strcmp(terms{1}{t}, '(')
                    % We have an actual word. Get its map index. (Beware: "index" is overloaded.)
                    wordIndex = wordIndex + 1;
                    p.wordIndices(wordIndex) = Pyramid.wordLookup(terms{1}{t}, wordMap);
                end
            end

            % Fill in the implicit connections in the tree
            for depth = 1:p.wordCount - 1
                seen = 0;
                for index = 1:depth
                    if p.connectionLabels(depth, index) == 3
                        seen = 1;
                    elseif seen == 0
                        p.connectionLabels(depth, index) = 1;
                    else
                        p.connectionLabels(depth, index) = 2;
                    end
                end
            end
        end

        
        function id = wordLookup(iText, wordMap)
            if wordMap.isKey(iText)
                id = wordMap(iText);
            elseif all(ismember(iText, '0123456789.-'))
                disp(['Collapsing number ' iText]);
                id = wordMap('<num>');      
            else
                % Account for possible use of exactAlign
                nextTry = strtok(iText, ':');
                if wordMap.isKey(nextTry)
                    id = wordMap(nextTry);
                % Try splitting hyphenated words
                elseif findstr('-', nextTry)
                    [first, remainder] = strtok(nextTry, '-');
                    converted = [first, ' ( - ', remainder(2:end), ' ) '];
                    t = Tree.makeTree(converted, wordMap);
                else
                    if wordMap.isKey('<unk>')
                        id = wordMap('<unk>');
                    else
                        assert(false, ['Failed to map word ' iText]);
                    end
                end
            end
        end
    end

    methods
        
        function t = getText(obj)
            t = obj.text;
        end

    end
end