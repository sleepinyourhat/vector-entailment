function dataset = Uninformativize(dataset)
% Strip out all training pairs but those with the '=' relation

dataset = dataset([dataset.relation] == 2);

end