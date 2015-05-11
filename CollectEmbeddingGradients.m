function embeddingGradients = CollectEmbeddingGradients(gradients, wordIndices, numEmbeddings)

B = size(gradients, 2);
M = zeros(B, numEmbeddings, 'like', gradients);
mIndices = sub2ind([B, numEmbeddings], 1:B, wordIndices);
M(mIndices) = 1;
embeddingGradients = gradients * M;

end
