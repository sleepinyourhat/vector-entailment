% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ wordMap, relationMap, relations, data ] = ...
    LoadWordPairData(filename, hyperParams)
% Load simple *word-word* pair data for pretraining and to generate a word map.

% For some experiments, this is only used to initialize the words and
% relations, and the data itself is not used.

'USING LWPD!'

% Load the file
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Initialize the data array
rawData = repmat(struct('relation', 0, 'leftText', '', 'rightText', ''), ...
    length(C{1}), 1);
wordList = cell(length(C{1}), 1);

% Establish (manually specified) relations
relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
relationMap = cell(1, 1);
relationMap{1} = containers.Map(relations{1}, 1:length(relations{1}));

% Parse the file
itemNo = 1;
wordNo = 1;
maxLine = length(C{1});

for line = 1:maxLine;
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', '\t');
        splitLine = splitLine{1};
        
        if ~(length(splitLine{1}) ~= 1 || splitLine{1} == '%')
            % Skip lines that are blank or have a multicharacter first chunk
            rawData(itemNo).relation = relationMap{1}(splitLine{1});
            rawData(itemNo).leftText = splitLine{2};
            rawData(itemNo).rightText = splitLine{3};

            % Add to word list
            lWords = textscan(splitLine{2}, '%s', 'delimiter', ' ');
            rWords = textscan(splitLine{3}, '%s', 'delimiter', ' ');
            words = unique([lWords{1}; rWords{1}]);
            wordList(wordNo:wordNo + (length(words) - 1)) = cellstr(words);
            wordNo = wordNo + length(words);

            itemNo = itemNo + 1;
        end
    end
end

rawData = rawData(1:itemNo - 1);

% Compile vocabulary
wordList = wordList(1:wordNo - 1);
vocabulary = unique(wordList);

% Remove syntactic symbols from the vocabulary
vocabulary = setdiff(vocabulary, {'(', ')'});

% Build word map
wordMap = containers.Map(vocabulary,1:length(vocabulary));

% This code isn't optimized for cases where we don't need the data itself, but
% here's a shortcut to avoid some work in those cases
if nargout > 3
    % Build the dataset
    data = repmat(struct('relation', 0, 'leftTree', Tree(), 'rightTree', Tree()), ...
        length(rawData), 1);

    % Build Trees
    for dataInd = 1:length(rawData)
        data(dataInd).leftTree = Tree.makeTree(rawData(dataInd).leftText, wordMap);
        data(dataInd).rightTree = Tree.makeTree(rawData(dataInd).rightText, wordMap);
        data(dataInd).relation = rawData(dataInd).relation;
    end
end

end

