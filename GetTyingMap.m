% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function tyingMap = GetTyingMap(wordMap)
% Create a map from word index to something like part of speech
% Q = 1
% QP = 2
% neg = 3
% predicate = 4

tyingMap = containers.Map(1:wordMap.Count,repmat(4, wordMap.Count, 1));

if isKey(wordMap, 'some')
    tyingMap(wordMap('some')) = 1;
    tyingMap(wordMap('all')) = 1;
    tyingMap(wordMap('most')) = 1;
    tyingMap(wordMap('no')) = 1;
    tyingMap(wordMap('two')) = 1;
    tyingMap(wordMap('three')) = 1;
    tyingMap(wordMap('not')) = 3;
end