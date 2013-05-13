function tensorLayerOutput = ComputeTensorLayer(combinedFeatures, classifierMatrices)
% Takes arguments: feature vector, matrix set

global DIM;
global NUM_RELATIONS;

% TODO: Do we want this one?

% NOTE: Currently this layer yields a vector arbitrarily
% limited to the length of the relation list.
tensorLayerOutput = zeros(NUM_RELATIONS,1);

for relNum = 1:NUM_RELATIONS
    Cols = (((DIM*2))*(relNum - 1))+1:(((DIM*2))*(relNum));
    tensorProd = combinedFeatures * classifierMatrices(:,Cols) * combinedFeatures';
    tensorLayerOutput(relNum) = Sigmoid(tensorProd);
end

end