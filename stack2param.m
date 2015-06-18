% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function varargout = stack2param(X, decodeInfo)
% Borrowed from Socher et al. Undoes param2stack.

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

assert(index==length(X))