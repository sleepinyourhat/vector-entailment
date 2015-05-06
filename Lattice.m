% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
classdef Lattice < handle
    % The object for loading and storing Lattices. NN computations are performed using LatticeBatch.
    
    properties
        wordIndices = [];  % The embedding matrix index for each word 
        wordCount = -1;  % The number of words in the sequence.
        connectionLabels = []  % optional N-1 x 1 vector indicating, for each row,
                               % which node at that row shuold do composition.
        legConnectionLabels = [];
        activeNode = [] % Lower triangular matrix of ones indicating which positions are part of the
                        % triangular lattice structure and which are just meaningless positions left 
                        % in to create a square matrix.
        text = 'NO TEXT'; % The sentence string.
    end
    
    methods(Static)
        function l = makeLattice(iText, wordMap, gpu, embGpu)
            l = Lattice();

            terms = textscan(iText, '%s', 'delimiter', ' ');

            % Parsed sequence case.
            assert(length(terms) == 1 || strncmpi(terms{1}{1}, '(', 1) && strncmpi(terms{1}{end}, ')', 1), ...
                   ['Input strings must be parsed, and must include the outermost parens: ', iText]);

            if length(terms{1}{1}) == 1
                % Normal parse tree mode
                l.wordCount = (length(terms{1}) + 2) / 3;  % Works for all binary parse trees.
            elseif length(terms{1}{1}) == 2 && strncmpi(terms{1}{1}, '(', 1)
                % SST mode
                l.wordCount = (length(terms{1}) + 2) / 5;  % Works for all binary parse trees with unary word wrappers.
            elseif length(terms) == 1
                l.wordCount = 1;
            else
                assert(false, ['Bad first element in parse string: ' iText]);
            end
                
            assert(~mod(l.wordCount, 1), ['Parse failure: ' iText])

            % TODO: Handle unparsed sequences.

            l.wordIndices = fZeros([l.wordCount, 1], embGpu);
            l.connectionLabels = fZeros([l.wordCount - 1, 1], gpu);
            l.activeNode = tril(fOnes([l.wordCount, l.wordCount], gpu), 0);
            l.text = iText;

            % Load the words and the tree structure.
            % TODO: Set up an option to treat parentheses as words as well as as connection supervision.
            depth = l.wordCount;
            mergeCount = 0;
            wordIndex = 0;  % The number of words that have been loaded.
            for t = 1:length(terms{1})
                % Mark the merge in the tree structure if this is a binary constituent.
                if strncmpi(terms{1}{t}, ')', 1) && ~strncmpi(terms{1}{t - 2}, '(', 1)
                    l.connectionLabels(depth - 1) = wordIndex - 1 - mergeCount;
                    depth = depth - 1;
                    mergeCount = mergeCount + 1;
                elseif ~strncmpi(terms{1}{t}, '(', 1) && ~strncmpi(terms{1}{t}, ')', 1)
                    % We have an actual word. Get its embedding index. (Beware: the word "index" is overloaded.)
                    wordIndex = wordIndex + 1;
                    l.wordIndices(wordIndex) = Lattice.wordLookup(terms{1}{t}, wordMap);
                end
            end
        end

        function id = wordLookup(iText, wordMap)
            % Get an embedding index for an arbitrary string.
            if wordMap.isKey(iText)
                id = wordMap(iText);
            elseif all(ismember(iText, '0123456789.-'))
                id = wordMap('<num>');      
            else
                nextTry = strtok(iText, ':');
                if wordMap.isKey(nextTry)
                    % Account for possible use of exactAlign.
                    id = wordMap(nextTry);
                    % TODO: Try splitting hyphenated words if they aren't in the dictionary whole.
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
