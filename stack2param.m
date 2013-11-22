function varargout = stack2param(X, decodeInfo)
% Borrowed from Socher et al

assert(length(decodeInfo)==nargout,'this should output as many variables as you gave to get X with param2stack!')

index=0;
for i=1:length(decodeInfo)
    if iscell(decodeInfo{i})
        for c = 1:length(decodeInfo{i})
            matSize = decodeInfo{i}{c};
            totalSize=prod(matSize);
            cellOut{c} = reshape(X(index+1:index+totalSize),matSize);
            index = index+totalSize;
        end
        varargout{i}=cellOut;
    else
        matSize = decodeInfo{i};
        totalSize=prod(matSize);
        varargout{i} = reshape(X(index+1:index+prod(matSize)),matSize);
        index = index+totalSize;
    end
end
% make sure, you used all params:
assert(index==length(X))