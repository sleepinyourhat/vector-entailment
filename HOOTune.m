for dim = [10 25 40]
    for mbs = [4 16 64 256]
        for tensor = [0 1]
            TrainJoinModel('test', dim, mbs, tensor)
            dim
            mbs
            tensor
        end
    end
end