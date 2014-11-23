function hyperParams = FlushLogs(hyperParams)
% Close and re-open the log files.
% This is attempting to circumvent a problem where logs from jobs logging to 
% AFS don't get refreshed on non-worker machines.

exampleFilename = fopen(hyperParams.examplelog);
statFilename = fopen(hyperParams.statlog);

fclose(hyperParams.examplelog);
fclose(hyperParams.statlog);

hyperParams.examplelog = fopen(exampleFilename, 'a');
hyperParams.statlog = fopen(statFilename, 'a');

end

