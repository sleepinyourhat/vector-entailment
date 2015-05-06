function result = fOnes(sizes, gpu)
% A more flexible 'zeros'
% Adapted from code by Thang Luong.

if gpu
    result = ones(sizes, 'single', 'gpuArray');  
else
    result = ones(sizes);
end

end
