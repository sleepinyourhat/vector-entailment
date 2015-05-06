function result = fNormrnd(m, s, sizes, gpu, singleOnly)
% A more flexible 'normrnd'
% Adapted from code by Thang Luong.

if singleOnly
    result = single(normrnd(m, s, sizes(1), sizes(2)));  
elseif gpu
	result = gpuArray(single(normrnd(m, s, sizes(1), sizes(2))));  
else
    result = normrnd(m, s, sizes);
end

end
