% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ wordMap, relationMap, relations ] = ...
    InitializeMaps(filename)
% Load a word map from text. For use with the SICK model setup.

relations = {'ENTAILMENT', 'CONTRADICTION', 'NEUTRAL', 'NONENTAILMENT'};
relationMap = containers.Map(relations,1:length(relations));

% Load the file
fid = fopen(filename);
C = textscan(fid,'%s','delimiter',sprintf('\n'));
fclose(fid);

% Load the word list
vocabulary = C{1};

% Build word map
wordMap = containers.Map(vocabulary,1:length(vocabulary));

end

