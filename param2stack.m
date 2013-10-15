function [stack decodeInfo] = param2stack(varargin)
% Borrowed from Socher et al 2013

stack = [];

for i=1:length(varargin)
    if iscell(varargin{i})
        for c = 1:length(varargin{i})
            decodeCell{c} = size(varargin{i}{c});
            stack = [stack ; varargin{i}{c}(:)];
        end
        decodeInfo{i} = decodeCell;
    else
        decodeInfo{i} = size(varargin{i});
        stack = [stack ; varargin{i}(:)];
    end
end