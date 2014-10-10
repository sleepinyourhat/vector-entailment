function Crossvalidate(varargin)

for i = 1:5
	TrainModel(varargin{:}, fold);
end

end
