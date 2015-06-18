% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ symmetrizedDataset ] = Symmetrize(dataset)
% Flips the pairwise order of elements in natural logic relation data and adjusts the
% relation labels to fit.

flippedDataset = dataset;
[flippedDataset(:).leftTree] = dataset(:).rightTree;
[flippedDataset(:).rightTree] = dataset(:).leftTree;

[flippedDataset([dataset(:).relation] == 3).relation] = deal(4);
[flippedDataset([dataset(:).relation] == 4).relation] = deal(3);
[flippedDataset([dataset(:).relation] == 2)] = [];

%  1:#     2:=     3:>     4:<     5:|     6:^     7:v

symmetrizedDataset = [dataset; flippedDataset];
    
end