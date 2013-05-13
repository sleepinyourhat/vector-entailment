% Constant model parameters:
global DIM;
DIM = 25;
global NUM_RELATIONS;
NUM_RELATIONS = 7;

% Import data
fid = fopen('demo_wordpairs.tsv');
C = textscan(fid, '%c %s %s');
fclose(fid);

% Learn vocabulary
vocabulary = unique([C{2}, C{3}]);

% Columns=words; Rows=dimensions
global wordFeatures;

% Randomly initialize word features. 
wordFeatures = normrnd(0, 1, length(vocabulary), DIM);

% Build word map
global wordMap;
wordMap = containers.Map(vocabulary,1:length(vocabulary));

% Establish (manually specified) relations
global relationMap;
relations = {'#', '=', '>', '<', '|', '^', 'v'};
relationMap = containers.Map(relations,1:length(relations));

data = repmat(struct('relation', 0, 'leftTree', Tree(), 'rightTree', Tree()), ...
    length(C{1}), 1);

% Build Trees
for wordInd = 1:length(C{1})
    leftWord = C{2}(wordInd);
    leftWord = leftWord{1};
    data(wordInd).leftTree = Tree.makeLeaf(leftWord);
    
    rightWord = C{3}(wordInd);
    rightWord = rightWord{1};
    data(wordInd).rightTree = Tree.makeLeaf(rightWord);
    
    relation = C{1}(wordInd);
    data(wordInd).relation = relationMap(relation);
end

global classifierParameters;

% Randomly initialize softmax layer.
classifierParameters = normrnd(0, 1, NUM_RELATIONS, NUM_RELATIONS);

global classifierMatrices;

% Randomly initialize tensor matrices.
classifierMatrices = abs(normrnd(0, 0.1, (DIM * 2) , ((DIM * 2) ) * NUM_RELATIONS));

% read in examples; parse to n x 3 cell array, with Lword, Rword, relation
% Lword and Rword should be tree objects.

TrainClassifierAndFeatures(data);

% pass examples to classifier with trained model
% report scores

% cf. ClassifyTrees

% read trees (richard's code?)

% build representations and model using classifywords