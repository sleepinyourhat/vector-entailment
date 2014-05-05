% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ data ] = LoadMeetJoinData(filename, wordMap)
% Load one file of constituent-pair data.

% Append data-4/ if we don't have a full path:
if isempty(strfind(filename, '/'))
    filename = ['join-algebra/', filename];
end
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the file

% Initialize the data array
rawData = repmat(struct('solution', 0, 'treeText', ''), ...
    length(C{1}), 1);

% Parse the file
itemNo = 1;
maxLine = length(C{1});
%maxLine = 25; 
%disp('TODO: FIX MAXLINE !!!!!!!!!!!!!!!')
for line = 1:maxLine
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', '\t');
        splitLine = splitLine{1};
        
        if ~(length(splitLine{1}) ~= 1 || splitLine{1} == '%')
            % Skip lines that are blank or have a multicharacter first chunk
            rawData(itemNo).solution = splitLine{1};
            rawData(itemNo).treeText = splitLine{2};

            itemNo = itemNo + 1;
        end
    end
end

rawData = rawData(1:itemNo - 1);

% Build the dataset
data = repmat(struct('solution', 0, 'tree', LatticeTree()), ...
    length(rawData), 1);

% Build Trees
for dataInd = 1:length(rawData)
    data(dataInd).tree = LatticeTree.makeTree(rawData(dataInd).treeText, wordMap);
    data(dataInd).solution = str2double(rawData(dataInd).solution) + 1;
end

end

