load('/afs/ir.stanford.edu/users/s/b/sbowman/theta-tr@472-deep-best-RNN.mat')
decoder = thetaDecoder
[classifierMatrices, classifierMatrix, classifierBias, ...
classifierParameters, wordFeatures, compositionMatrices,...
compositionMatrix, compositionBias, classifierExtraMatrix, ...
classifierExtraBias] ...
= stack2param(theta, decoder);




iter2{1} = Tree.makeTree('oona', wordMap)
iter2{2} = Tree.makeTree('ollie ( or oona )', wordMap)
iter2{3} = Tree.makeTree('ollie ( or ( ollie ( or oona ) ) )', wordMap)
iter2{4} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or oona ) ) ) ) )', wordMap)
iter2{5} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or oona ) ) ) ) ) ) )', wordMap)
iter2{6} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or oona ) ) ) ) ) ) ) ) )', wordMap)
iter2{7} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or oona ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter2{8} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or oona ) ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter2{9} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or oona ) ) ) ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter2{10} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or oona ) ) ) ) ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter2{11} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or oona ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter2{12} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or oona ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter2{13} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or oona ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )', wordMap)



iter{1} = Tree.makeTree('ollie', wordMap)
iter{2} = Tree.makeTree('ollie ( or ollie )', wordMap)
iter{3} = Tree.makeTree('ollie ( or ( ollie ( or ollie ) ) )', wordMap)
iter{4} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ollie ) ) ) ) )', wordMap)
iter{5} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ollie ) ) ) ) ) ) )', wordMap)
iter{6} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ollie ) ) ) ) ) ) ) ) )', wordMap)
iter{7} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ollie ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter{8} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ollie ) ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter{9} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ollie ) ) ) ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter{10} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ollie ) ) ) ) ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter{11} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ollie )  ) ) ) ) ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter{12} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ollie ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )', wordMap)
iter{13} = Tree.makeTree('ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ( ollie ( or ollie ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )', wordMap)



errors = zeros(1, 8)
features = zeros(1,25)
features2 = zeros(1,25)
for i = 1:13
    iter{i}.updateFeatures(wordFeatures, compositionMatrices, compositionMatrix, compositionBias, @Sigmoid)
    iter2{i}.updateFeatures(wordFeatures, compositionMatrices, compositionMatrix, compositionBias, @Sigmoid)

    features(i, :) = iter{i}.features;
    features2(i, :) = iter2{i}.features;
end

for i = 1:13
        errors(i) = norm(features(i,:) - features2(i,:), 2);
end




iter{1} = Tree.makeTree('ollie', wordMap)
iter{2} = Tree.makeTree('not ollie', wordMap)
iter{3} = Tree.makeTree('not ( not ollie )', wordMap)
iter{4} = Tree.makeTree('not ( not ( not ollie ) )', wordMap)
iter{5} = Tree.makeTree('not ( not ( not ( not ollie ) ) )', wordMap)
iter{6} = Tree.makeTree('not ( not ( not ( not ( not ollie ) ) ) )', wordMap)
iter{7} = Tree.makeTree('not ( not ( not ( not ( not ( not ollie ) ) ) ) )', wordMap)
iter{8} = Tree.makeTree('not ( not ( not ( not ( not ( not ( not ollie ) ) ) ) ) )', wordMap)
iter{9} = Tree.makeTree('not ( not ( not ( not ( not ( not ( not ( not ollie ) ) ) ) ) ) )', wordMap)
iter{10} = Tree.makeTree('not ( not ( not ( not ( not ( not ( not ( not ( not ollie ) ) ) ) ) ) ) )', wordMap)
iter{11} = Tree.makeTree('not ( not ( not ( not ( not ( not ( not ( not ( not ( not ollie ) ) ) ) ) ) ) ) )', wordMap)
iter{12} = Tree.makeTree('not ( not ( not ( not ( not ( not ( not ( not ( not ( not ( not ollie ) ) ) ) ) ) ) ) ) )', wordMap)


errands = zeros(1, 8)

fand i = 1:12
    iter{i}.updateFeatures(wordFeatures, compositionMatrices, compositionMatrix, compositionBias, @Sigmoid)
    features(i, :) = iter{i}.features;
    errands(i) = nandm(features(1,:) - features(i,:), 2)
    
end
