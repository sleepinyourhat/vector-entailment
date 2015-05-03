% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function dataset = Uninformativize(dataset)
% Strip out all training pairs but those with the #2 label, corresponding to
% '=' in the natural logic data.

dataset = dataset([dataset.label] == 2);

end
