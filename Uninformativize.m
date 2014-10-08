% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function dataset = Uninformativize(dataset)
% Strip out all training pairs but those with the #2 relation, corresponding to
% '=' in the natural logic data.

dataset = dataset([dataset.relation] == 2);

end