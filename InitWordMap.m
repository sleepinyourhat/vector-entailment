function success = InitWords()

load("data/vars.normalized.100.mat")

global wordsToIndicesMap;
wordsToIndicesMap = containers.Map(words,1:length(words));