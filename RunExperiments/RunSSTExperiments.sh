cd quant; 
lambda = 0.000001; dim = 45; ed = 50; td = 2; penult = 75; dropout = [0.5, 0.5]; tot = 2; collo = 2; dataflag='snli095-only'; name='/scr/sbowman/sst-test'; 
TrainModel('', 1, @SST, name, dataflag, ed, dim, td, penult, lambda, tot, dropout(1), dropout(2), collo, 1, 1)
