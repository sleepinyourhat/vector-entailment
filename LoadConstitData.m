function [ data ] = LoadConstitData(filename, wordMap, relationMap)

% Append data/ if we don't have a full path:
if isempty(strfind(filename, '/'))
    filename = ['data-2/', filename];
end
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the file

% Initialize the data array
rawData = repmat(struct('relation', 0, 'leftText', '', 'rightText', ''), ...
    length(C{1}), 1);
% wordList = cell(length(C{1}), 1);

% Parse the file
itemNo = 1;
maxLine = length(C{1});
for line = 1:maxLine
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', '\t');
        splitLine = splitLine{1};
        
        if ~(length(splitLine{1}) ~= 1 || splitLine{1} == '%')
            % Skip lines that are blank or have a multicharacter first chunk
            rawData(itemNo).relation = relationMap(splitLine{1});
            rawData(itemNo).leftText = splitLine{2};
            rawData(itemNo).rightText = splitLine{3};

            itemNo = itemNo + 1;
        end
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

