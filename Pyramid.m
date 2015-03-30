% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef Pyramid < handle
    % The object for loading and storing Pyramids. NN computations are performed using PyramidBatch.
    
    properties
        wordIndices = [];  % The embedding matrix index for each word 
        wordCount = -1;  % The number of words in the sequence.
        connectionLabels = []  % optional matrix of correct connection types at each position
                                 % 1 := Copy left child.
                                 % 2 := Copy right child.
                                 % 3 := Compose left and right children.
        activeNode = [] % Lower triangular matrix of ones indicating which positions are part of the
                        % triangular pyramid structure and which are just meaningless positions left 
                        % in to create a square matrix.
        text = 'NO TEXT'; % The sentence string.
    end
    
    methods(Static)
        function p = makePyramid(iText, wordMap)
            p = Pyramid();

            terms = textscan(iText, '%s', 'delimiter', ' ');

            % Parsed sequence case.
            assert(length(terms) == 1 || strcmp(terms{1}{1}, '(') && strcmp(terms{1}{end}, ')'), ...
                   'Input strings must be parsed, and must include the outermost parens.');
            p.wordCount = (length(terms{1}) + 2) / 3;  % Works for all binary parse trees.

            % TODO: Handle unparsed sequences.

            p.wordIndices = zeros(p.wordCount, 1);
            p.connectionLabels = zeros(p.wordCount - 1, p.wordCount - 1);
            p.activeNode = zeros(p.wordCount, p.wordCount);
            p.text = iText;

            % Load the words and the tree structure.
            % TODO: Set up an option to treat parentheses as words as well as as connection supervision.
            depth = p.wordCount;
            mergeCount = 0;
            wordIndex = 0;  % The number of words that have been loaded.
            for t = 1:length(terms{1})
                if strcmp(terms{1}{t}, ')')  % Mark the merge in the tree structure.
                    p.connectionLabels(depth - 1, wordIndex - 1 - mergeCount) = 3;
                    depth = depth - 1;
                    mergeCount = mergeCount + 1;
                elseif ~strcmp(terms{1}{t}, '(')
                    % We have an actual word. Get its embedding index. (Beware: the word "index" is overloaded.)
                    wordIndex = wordIndex + 1;
                    p.wordIndices(wordIndex) = Pyramid.wordLookup(terms{1}{t}, wordMap);
                end
            end

            % Fill all of the connection labels.
            for depth = 1:p.wordCount
                seen = 0;
                for index = 1:depth
                    if depth < p.wordCount
                        if p.connectionLabels(depth, index) == 3
                            seen = 1;
                        elseif seen == 0
                            p.connectionLabels(depth, index) = 1;
                        else
                            p.connectionLabels(depth, index) = 2;
                        end
                    end
                    p.activeNode(depth, index) = 1;
                end
            end
        end

        function id = wordLookup(iText, wordMap)
            % Get an embedding index for an arbitrary string.
            if wordMap.isKey(iText)
                id = wordMap(iText);
            elseif all(ismember(iText, '0123456789.-'))
                disp(['Collapsing number ' iText]);
                id = wordMap('<num>');      
            else
                nextTry = strtok(iText, ':');
                if wordMap.isKey(nextTry)
                    % Account for possible use of exactAlign.
                    id = wordMap(nextTry);
                elseif findstr('-', nextTry)
                    % Try splitting hyphenated words if they aren't in the dictionary whole.
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
        
        % TODO: Connection visualization tools.
    end
end
