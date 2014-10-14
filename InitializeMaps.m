% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ wordMap, relationMap, relations ] = ...
    InitializeMaps(filename, dataflag)
% Load a word map from text. For use with the SICK model setup.

if findstr(dataflag, 'sick-') 
	relations = {{'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL'}, {'ENTAILMENT', 'NONENTAILMENT'}};
	relationMap = cell(2, 1);
	relationMap{1} = containers.Map(relations{1}, 1:length(relations{1}));
	relationMap{2} = containers.Map(relations{2}, 1:length(relations{2}));
elseif strcmp(dataflag, 'imageflickr') 
	relations = {{'ENTAILMENT', 'null1', 'null2', 'NONENTAILMENT'}};
	relationMap = cell(1, 1);
	relationMap{1} = containers.Map(relations{1}, 1:length(relations{1}));
elseif findstr(dataflag, 'G-')
	relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
	relationMap = cell(1, 1);
	relationMap{1} = containers.Map(relations{1}, 1:length(relations{1}));
elseif findstr(dataflag, 'gradcheck')
	relations = {{'#', '=', '>', '<', '|', '^', 'v'}};
	relationMap = cell(1, 1);
	relationMap{1} = containers.Map(relations{1}, 1:length(relations{1}));
elseif findstr(dataflag, 'synset')
	relations = {{'hypernym', 'hyponym', 'coordinate'}};
	relationMap = cell(1, 1);
	relationMap{1} = containers.Map(relations{1}, 1:length(relations{1}));
else 
	relations = {{'antonym', 'hypernym', 'hyponym', 'synonym'}};
	relationMap = cell(1, 1);
	relationMap{1} = containers.Map(relations{1}, 1:length(relations{1}));
end

% Load the file
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the word list
vocabulary = C{1};

% Build word map
wordMap = containers.Map(vocabulary,1:length(vocabulary));

end

