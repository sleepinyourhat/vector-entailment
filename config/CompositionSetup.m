function [ hyperParams ] = CompositionSetup(hyperParams, composition)
% Set up the various composition function configurations.
% Called by other config files.

if composition == -2
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 1;
	hyperParams.useSumming = 1;
elseif composition == -1
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 0;
	hyperParams.useSumming = 1;
elseif composition < 2
	hyperParams.useThirdOrderComposition = composition;
	hyperParams.useThirdOrderMerge = 1;
elseif composition == 2
	hyperParams.lstm = 1;
	hyperParams.useTrees = 0;
	hyperParams.eyeScale = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 1;
elseif composition == 3
	hyperParams.lstm = 0;
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 1;
elseif composition == 4
	hyperParams.useLattices = 1;
	hyperParams.lstm = 0;
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 0;
elseif composition == 5
	hyperParams.useLattices = 1;
	hyperParams.lstm = 0;
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 0;
elseif composition == 6
	hyperParams.useLattices = 1;
	hyperParams.lstm = 1;
	hyperParams.eyeScale = 0;
	hyperParams.useTrees = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 0;
elseif composition == 7
	hyperParams.lstm = 1;
	hyperParams.useTrees = 1;
	hyperParams.eyeScale = 0;
	hyperParams.useThirdOrderComposition = 0;
	hyperParams.useThirdOrderMerge = 1;
end

end
