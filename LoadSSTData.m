% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function data = LoadSSTData(filename, wordMap, labelMap, hyperParams, fragment, labelSetIndex)
% Load one file of sentence pair data.

if hyperParams.useTrees
    typeSig = '-trees';
elseif hyperParams.useLattices
    typeSig = '-lats';
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
maxLine = min(length(C{1}), hyperParams.lineLimit);
% maxLine = min(35, maxLine);

% Initialize the data array
rawData = repmat(struct('label', 0, 'sentenceText', ''), maxLine, 1);

% Which nextItemNo was the last to be included in the last MAT file.
lastSave = 0;

% Iterate over examples
for line = (lastSave + 1):maxLine
    if ~isempty(C{1}{line}) 

        % Skip commented and unlabeled lines
        if (C{1}{line}(1) == '(') && labelMap{labelSetIndex}.isKey(C{1}{line}(2))
            rawData(nextItemNo - lastSave).label = [ labelMap{labelSetIndex}(C{1}{line}(2)); labelSetIndex ];
            rawData(nextItemNo - lastSave).sentenceText = C{1}{line};
            nextItemNo = nextItemNo + 1;
        else
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
        data = repmat(struct('label', 0, 'sentence', Tree()), numElements, 1);
        parfor dataInd = 1:numElements
            data(dataInd).sentence = Tree.makeTree(rawData(dataInd).sentenceText, wordMap);
            data(dataInd).label = rawData(dataInd).label;
        end
    elseif hyperParams.useLattices
        data = repmat(struct('label', 0, 'sentence', Lattice()), numElements, 1);
        parfor dataInd = 1:numElements
            data(dataInd).sentence = Lattice.makeLattice(rawData(dataInd).sentenceText, wordMap, hyperParams.gpu, ...
                hyperParams.gpu && ~hyperParams.largeVocabMode);
            data(dataInd).label = rawData(dataInd).label;
        end
    else
        data = repmat(struct('label', 0, 'sentence', Sequence()), numElements, 1);
        for dataInd = 1:numElements
            data(dataInd).sentence = Sequence.makeSequence(rawData(dataInd).sentenceText, wordMap, ...
                hyperParams.parensInSequences, hyperParams.gpu && ~hyperParams.largeVocabMode);
            data(dataInd).label = rawData(dataInd).label;
        end
    end

    if ~hyperParams.ignorePreprocessedFiles
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
end
