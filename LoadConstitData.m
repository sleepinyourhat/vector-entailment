% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ data ] = LoadConstitData(filename, wordMap, relationMap, hyperParams, fragment, relationIndex)
% Load one file of sentence pair data.

% Append a default prefix if we don't have a full path
if isempty(strfind(filename, '/'))
    if strfind(filename, 'quant_')
        filename = ['grammars/data/', filename];
    end
end

if fragment
    % Check whether we already loaded this file
    [pathname, filenamePart, ext] = fileparts(filename);
    listing = dir([pathname, '/pp-', filenamePart, ext, '-final-', hyperParams.vocabName, '*']);
    if length(listing) > 0
        Log(hyperParams.statlog, ['File ', filename, ' was already processed.']);
        return
    end
else
    % Check whether we already loaded this file
    [pathname, filenamePart, ext] = fileparts(filename);
    listing = dir([pathname, '/pp-', filenamePart, ext, '-full-', hyperParams.vocabName, '*']);
    if length(listing) > 0
        Log(hyperParams.statlog, ['File ', filename, ' was already processed. Loading.']);
        d = load(listing(1).name,'-mat');
        data = d.data;
        return
    end
end

fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Initialize the data array
rawData = repmat(struct('relation', 0, 'leftText', '', 'rightText', ''), ...
    length(10000), 1);

% Parse the file
nextItemNo = 1;
maxLine = length(C{1});

% Which nextItemNo was the last to be included in the last MAT file.
lastSave = 0;

% Iterate over examples
for line = (lastSave + 1):maxLine
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', '\t');
        splitLine = splitLine{1};
        
        if (sum(splitLine{1} == '%') == 0) && (size(splitLine, 1) >= 3)
            % Skip commented lines
            if nargin > 5
                rawData(nextItemNo - lastSave).relation = zeros(length(hyperParams.numRelations), 1);
                rawData(nextItemNo - lastSave).relation(relationIndex) = ...
                    relationMap{relationIndex}(splitLine{1});
            else
                rawData(nextItemNo - lastSave).relation = relationMap(splitLine{1});
            end
            rawData(nextItemNo - lastSave).leftText = splitLine{2};
            rawData(nextItemNo - lastSave).rightText = splitLine{3};
            nextItemNo = nextItemNo + 1;
        else
            disp('Skipped line.');
        end
    end
    if (mod(nextItemNo - 1, 10000) == 0 && nextItemNo > 0 && fragment)
        message = ['Lines loaded: ', num2str(nextItemNo), '/~', num2str(maxLine)];
        Log(hyperParams.statlog, message);
        data = ProcessAndSave(rawData, wordMap, lastSave, nextItemNo, filename, hyperParams, fragment);
        lastSave = nextItemNo - 1;
    end
end

if fragment
    data = ProcessAndSave(rawData, wordMap, lastSave, nextItemNo, [filename, '-final'], hyperParams, fragment);
else
    data = ProcessAndSave(rawData, wordMap, lastSave, nextItemNo, [filename, '-full'], hyperParams, fragment);
end
    

end

function [ data ] = ProcessAndSave(rawData, wordMap, lastSave, nextItemNo, filename, hyperParams, fragment)
    numElements = nextItemNo - (lastSave + 1);

    if hyperParams.useTrees
        data = repmat(struct('relation', 0, 'left', Tree(), 'right', Tree()), numElements, 1);
        parfor dataInd = 1:numElements
            data(dataInd).left = Tree.makeTree(rawData(dataInd).leftText, wordMap);
            data(dataInd).right = Tree.makeTree(rawData(dataInd).rightText, wordMap);
            data(dataInd).relation = rawData(dataInd).relation;
        end
    else
        data = repmat(struct('relation', 0, 'left', Sequence(), 'right', Sequence()), numElements, 1);
        parfor dataInd = 1:numElements
            data(dataInd).left = Sequence.makeSequence(rawData(dataInd).leftText, wordMap, hyperParams.parensInSequences);
            data(dataInd).right = Sequence.makeSequence(rawData(dataInd).rightText, wordMap, hyperParams.parensInSequences);
            data(dataInd).relation = rawData(dataInd).relation;
        end
    end

    % Turn on if further debugging is needed. Slow.
    % for i = 1:length(data)
    %     assert(~isempty(data(i).left.getText()), ['Did not finish processing.' num2str(i)]);
    %     assert(~isempty(data(i).right.getText()), ['Did not finish processing.' num2str(i)]);       
    % end

    if fragment
        [pathname, filenamePart, ext] = fileparts(filename);
        save([pathname, '/pp-', filenamePart, ext, '-', hyperParams.vocabName, '-', num2str(nextItemNo), '.mat'], 'data');
    end
end
