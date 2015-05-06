function result = fRand(sizes, gpu)
% A more flexible 'rand'
% Adapted from code by Thang Luong.

if gpu
    result = rand(sizes, 'single', 'gpuArray');  
else
    result = rand(sizes);
end

end
