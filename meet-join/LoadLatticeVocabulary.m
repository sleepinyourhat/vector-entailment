% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ wordMap ] = ...
    LoadLatticeVocabulary(filename)
% Load word-word pair data for pretraining and to generate a word map.

% For some experiments, this is only used to initialize the words and
% relations, and the data itself is not used.


% Load the file
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Initialize the data array
rawData = repmat(struct('solution', 0, 'treeText', ''), ...
    length(C{1}), 1);

% Parse the file
itemNo = 1;
maxWord = -1;
maxLine = length(C{1});
% maxLine = 10; % Uncomment to truncate data for testing.

for line = 1:maxLine;
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', '\t');
        splitLine = splitLine{1};
        
        if ~(length(splitLine{1}) ~= 1 || splitLine{1} == '%')
            % Skip lines that are blank or have a multicharacter first chunk
            rawData(itemNo).solution = str2double(splitLine{1});
            rawData(itemNo).treeText = splitLine{2};

            % Note highest word index
            if rawData(itemNo).solution > maxWord
                maxWord = rawData(itemNo).solution;
            end

            itemNo = itemNo + 1;
        end
    end
end

rawData = rawData(1:itemNo - 1);

words = cell(maxWord + 1, 1);
for i = 1:maxWord + 1
    words{i} = num2str(i - 1);
end

% Build word map
wordMap = containers.Map(words, 1:maxWord + 1);

% This isn't optimized for cases where we don't need the data itself, this 
% is just a shortcut:
if nargout > 3
    % Build the dataset
    data = repmat(struct('solution', 0, 'tree', LatticeTree()), ...
        length(rawData), 1);

    % Build Trees
    for dataInd = 1:length(rawData)
        data(dataInd).tree = LatticeTree.makeTree(rawData(dataInd).treeText, wordMap);
        data(dataInd).solution = rawData(dataInd).solution;
    end
    % data = [data; Symmetrize(data)];
end

end

