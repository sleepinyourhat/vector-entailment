function [ flippedDataset ] = Symmetrize(dataset)
% Flips the order of elements in an NL relation dataset and adjusts the
% relation class to fit.

flippedDataset = dataset
[flippedDataset(:).leftTree] = dataset(:).rightTree;
[flippedDataset(:).rightTree] = dataset(:).leftTree;

%  1:#     2:=     3:>     4:<     5:|     6:^     7:v

parfor i = 1:length(dataset) % Vectorize
    if dataset(i).relation == 3 % #
        flippedDataset(i).relation = 4;
    elseif dataset(i).relation == 4 % 
        flippedDataset(i).relation = 3;
    else 
end
    
end