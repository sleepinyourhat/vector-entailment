function Log(file, message)
% Write a message to a log file with a timestamp, and optionally 
% display it as well.

if file < 3
	disp(['Bad FileID: ' file]);
else
	disp(message);
	try
		fprintf(file, '%s\n', [datestr(clock, 0) ': ' message]);
	catch err
	end
end

end
