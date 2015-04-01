% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function data = LoadConstitData(filename, wordMap, relationMap, hyperParams, fragment, relationIndex)
% Load one file of sentence pair data.

if hyperParams.useTrees
    typeSig = '-trees';
elseif hyperParams.usePyramids
    typeSig = '-pyrs';
else
    typeSig = ['-seqs-par' num2str(hyperParams.parensInSequences)];
end
    
if fragment
    % Check whether we already loaded this file
    [pathname, filenamePart, ext] = fileparts(filename);
    listing = dir([pathname, '/pp-', filenamePart, ext,'-final-', hyperParams.vocabName, typeSig, '*']);
    if length(listing) > 0
        Log(hyperParams.statlog, ['File ', filename, ' was already processed.']);
        return
    end
elseif ~hyperParams.ignorePreprocessedFiles
    % Check whether we already loaded this file
    [pathname, filenamePart, ext] = fileparts(filename);
    listing = dir([pathname, '/pp-', filenamePart, ext, '-full-', hyperParams.vocabName, typeSig, '*']);
    if length(listing) > 0 
        Log(hyperParams.statlog, ['File ', filename, ' was already processed. Loading.']);
        try
            d = load([pathname, '/', listing(1).name],'-mat');
            data = d.data;
            return
        catch
            Log(hyperParams.statlog, 'Problem loading preprocessed data. Will reprocess raw file.');
        end
    end
end

fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Parse the file
nextItemNo = 1;
maxLine = length(C{1});

% Initialize the data array
rawData = repmat(struct('relation', 0, 'leftText', '', 'rightText', ''), maxLine, 1);

% Which nextItemNo was the last to be included in the last MAT file.
lastSave = 0;

% Iterate over examples
for line = (lastSave + 1):maxLine
    if ~isempty(C{1}{line}) 
        splitLine = textscan(C{1}{line}, '%s', 'delimiter', '\t');
        splitLine = splitLine{1};
        
        % Skip commented and unlabeled lines
        if (splitLine{1}(1) ~= '%') && (splitLine{1}(1) ~= '-') && (size(splitLine, 1) >= 3)
            if nargin > 5
                rawData(nextItemNo - lastSave).relation = [ relationMap{relationIndex}(splitLine{1}); relationIndex ];
            else
                rawData(nextItemNo - lastSave).relation = relationMap(splitLine{1});
            end
            rawData(nextItemNo - lastSave).leftText = splitLine{2};
            rawData(nextItemNo - lastSave).rightText = splitLine{3};
            nextItemNo = nextItemNo + 1;
        elseif splitLine{1}(1) ~= '-'
            disp(['Skipped line: ' C{1}{line}]);
        end
    end
    if (mod(nextItemNo - 1, 10000) == 0 && nextItemNo > 0 && fragment)
        message = ['Lines loaded: ', num2str(nextItemNo), '/~', num2str(maxLine)];
        Log(hyperParams.statlog, message);
        data = ProcessAndSave(rawData, wordMap, lastSave, nextItemNo, filename, hyperParams, fragment, typeSig);
        lastSave = nextItemNo - 1;
    end
end

if fragment
    data = ProcessAndSave(rawData, wordMap, lastSave, nextItemNo, [filename, '-final'], hyperParams, fragment, typeSig);
else
    data = ProcessAndSave(rawData, wordMap, lastSave, nextItemNo, [filename, '-full'], hyperParams, fragment, typeSig);
end
    
end

function [ data ] = ProcessAndSave(rawData, wordMap, lastSave, nextItemNo, filename, hyperParams, fragment, typeSig)
    numElements = nextItemNo - (lastSave + 1);

    if hyperParams.useTrees
        data = repmat(struct('relation', 0, 'left', Tree(), 'right', Tree()), numElements, 1);
        parfor dataInd = 1:numElements
            data(dataInd).left = Tree.makeTree(rawData(dataInd).leftText, wordMap);
            data(dataInd).right = Tree.makeTree(rawData(dataInd).rightText, wordMap);
            data(dataInd).relation = rawData(dataInd).relation;
        end
    elseif hyperParams.usePyramids
        data = repmat(struct('relation', 0, 'left', Pyramid(), 'right', Pyramid()), numElements, 1);
        parfor dataInd = 1:numElements
            data(dataInd).left = Pyramid.makePyramid(rawData(dataInd).leftText, wordMap);
            data(dataInd).right = Pyramid.makePyramid(rawData(dataInd).rightText, wordMap);
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

    [pathname, filenamePart, ext] = fileparts(filename);
    nameToSave = [pathname, '/pp-', filenamePart, ext, '-', hyperParams.vocabName, typeSig, '-', num2str(nextItemNo), '.mat'];
    listing = dir(nameToSave);
    % Double check that a file hasn't been written while we were processing.
    if isempty(listing)
        try
            save(nameToSave, 'data', '-v7.3');
        catch
            Log(hyperParams.statlog, 'Problem saving.');
        end
    end
end
