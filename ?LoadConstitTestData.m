function [ data ] = LoadConstitData(filename, wordMap, relationMap)
%[ dataset ] = LoadTrainingData()

% 'constitpairs-v1.tsv'

% Import data
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the file

% Initialize the data array
rawData = repmat(struct('relation', 0, 'leftText', '', 'rightText', ''), ...
    length(C{1}), 1);
wordList = cell(length(C{1}), 1);

% Parse the file
itemNo = 1;
wordNo = 1;
maxLine = length(C{1});
% maxLine = 6; % Truncate data.
for line = 1:maxLine;
    splitLine = strsplit(C{1}{line}, '\t');
    if ~(isempty(splitLine) || length(splitLine{1}) ~= 1 || splitLine{1} == '%')
        % Skip lines that are blank or have a multicharacter first chunk
        rawData(itemNo).relation = relationMap(splitLine{1});
        rawData(itemNo).leftText = splitLine{2};
        rawData(itemNo).rightText = splitLine{3};
        
        % Add to wordList
        words = unique([strsplit(splitLine{2}), strsplit(splitLine{3})]);
        wordList(wordNo:wordNo + (length(words) - 1)) = cellstr(words);
        wordNo = wordNo + length(words);
        
        itemNo = itemNo + 1;
    end
end

rawData = rawData(1:itemNo - 1);

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

