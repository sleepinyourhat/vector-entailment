% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cutDataset ] = Asymmetrize(dataset)
% Assumes a symmetric input dataset of word pairs where iff (a,b) is in the
% dataset, (b,a) is too. Removes this property by deleting half of the 
% examples where a ~= b.

% Preallocate based on a hacky optimistic approximation of the size.
cutDataset = repmat(struct('label', 0, 'leftTree', Tree(), 'rightTree', Tree()), ...
    length(dataset) * .6, 1);

j = 1;
for i = 1:length(dataset)
    if dataset(i).leftTree.getWordIndex() >= ...
       dataset(i).rightTree.getWordIndex()
        cutDataset(j) = dataset(i);
        j = j + 1;
    end
end
    
end
