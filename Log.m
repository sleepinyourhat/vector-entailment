function Log(file, message)
% Write a message to a log file with a timestamp, and optionally 
% display it as well.

% TODO: Set up non-verbose mode
disp(message);
fprintf(file, '%s\n', [datestr(clock, 0) ': ' message]);

end