% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function hyperParams = FlushLogs(hyperParams)
% Close and re-open the log files.
% TODO: There may be some places left in the code where the old FID might persist. Debug.

% This works around a problem where logs on AFS don't get refreshed
% on non-worker machines until the job finishes.

exampleFilename = fopen(hyperParams.examplelog);
statFilename = fopen(hyperParams.statlog);

fclose(hyperParams.examplelog);
fclose(hyperParams.statlog);

hyperParams.examplelog = fopen(exampleFilename, 'a');
hyperParams.statlog = fopen(statFilename, 'a');

end

