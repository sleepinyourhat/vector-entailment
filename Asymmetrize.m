% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cutDataset ] = Asymmetrize(dataset)
% Assumes a symmetric input dataset of word pairs where iff (a,b) is in the
% dataset, (b,a) is too. Removes this property by deleting half of the 
% examples where a ~= b.

% Preallocate approximate size
cutDataset = repmat(struct('relation', 0, 'leftTree', Tree(), 'rightTree', Tree()), ...
    length(dataset)/2, 1);

j = 1;
for i = 1:length(dataset)
    if dataset(i).leftTree.getWordIndex() >= ...
       dataset(i).rightTree.getWordIndex()
        cutDataset(j) = dataset(i);
        j = j + 1;
        % disp([dataset(i).leftTree.getText(), dataset(i).rightTree.getText()]);
    end
end
    
end