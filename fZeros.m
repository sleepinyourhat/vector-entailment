function result = fZeros(sizes, gpu)
% A more flexible 'zeros'
% Adapted from code by Thang Luong.

if gpu
    result = gpuArray(zeros(sizes, 'single'));  
else
    result = zeros(sizes);
end

end
