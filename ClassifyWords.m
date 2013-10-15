% Constant model parameters:
global DIM;
DIM = 25;
global NUM_RELATIONS;
NUM_RELATIONS = 7;
global PENULT_DIM;
PENULT_DIM = 10;


% Import data
fid = fopen('demo_wordpairs.tsv');
C = textscan(fid, '%c %s %s');
fclose(fid);

% Learn vocabulary
vocabulary = unique([C{2}, C{3}]);

% Columns=words; Rows=dimensions
global wordFeatures;

% Randomly initialize word features. 
wordFeatures = rand(length(vocabulary), DIM) .* .02 - .01;

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
classifierParameters = rand(NUM_RELATIONS, PENULT_DIM + 1) .* .02 - .01;

global classifierMatrices;

% Randomly initialize tensor matrices.
classifierMatrices = rand(DIM , (DIM * PENULT_DIM)) .* .02 - .01;

global classifierMatrix;

classifierMatrix = rand(PENULT_DIM, DIM * 2) .* .02 - .01;

global classifierBias;

classifierBias = rand(PENULT_DIM, 1) .* .02 - .01;

% read in examples; parse to n x 3 cell array, with Lword, Rword, relation
% Lword and Rword should be tree objects.

TrainClassifierAndFeatures(data);

% pass examples to classifier with trained model
% report scores

% cf. ClassifyTrees

% read trees (richard's code?)

% build representations and model using classifywords