function [ data, wordMap, relationMap ] = LoadConstitTrainingData(filename)
%[ dataset ] = LoadTrainingData()

% 'constitpairs-v1.tsv'

% Import data
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);


% TRUNCATE
%C{1} = C{1}(1:5)
%C{2} = C{2}(1:5)
%C{3} = C{3}(1:5)

% Load the file

% Initialize the data array
rawData = repmat(struct('relation', 0, 'leftText', '', 'rightText', ''), ...
    length(C{1}), 1);
wordList = cell(length(C{1}), 1);

% Establish (manually specified) relations
relations = {'#', '=', '>', '<', '|', '^', 'v'};
relationMap = containers.Map(relations,1:length(relations));

% Parse the file
itemNo = 1;
wordNo = 1;
maxLine = length(C{1});
% maxLine = 6; % Truncate data.

for line = 1:maxLine;
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', '\t');
        splitLine = splitLine{1};
        
        if ~(length(splitLine{1}) ~= 1 || splitLine{1} == '%')
            % Skip lines that are blank or have a multicharacter first chunk
            rawData(itemNo).relation = relationMap(splitLine{1});
            rawData(itemNo).leftText = splitLine{2};
            rawData(itemNo).rightText = splitLine{3};

            % Add to wordList
            lWords = textscan(splitLine{1}, '%s', 'delimiter', ' ');
            rWords = textscan(splitLine{2}, '%s', 'delimiter', ' ');
            words = unique([lWords{1}, rWords{1}]);
            wordList(wordNo:wordNo + (length(words) - 1)) = cellstr(words);
            wordNo = wordNo + length(words);

            itemNo = itemNo + 1;
        end
    end
end

rawData = rawData(1:itemNo - 1);

% Learn vocabulary
wordList = wordList(1:wordNo - 1);
vocabulary = unique(wordList);

% Remove syntactic symbols from vocabulary
vocabulary = setdiff(vocabulary, {'(', ')'});

% Build word map
wordMap = containers.Map(vocabulary,1:length(vocabulary));

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

