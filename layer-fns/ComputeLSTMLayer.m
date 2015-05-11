% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ h, c, IFOGf ] = ComputeLSTMLayer(WLSTM, h_prev, c_prev, x)
% Compute the LSTM activation for a batch of inputs.

% Based on an implementation by A. Karpathy here:
% https://github.com/karpathy/neuraltalk/blob/master/imagernn/lstm_generator.py

[ D, B ] = size(x);

Ir = 0 * D + 1:1 * D;
Fr = 1 * D + 1:2 * D;
Or = 2 * D + 1:3 * D;
Gr = 3 * D + 1:4 * D;

in = [ones(1, size(x, 2), 'like', x); x; h_prev];

IFOG = WLSTM * in;

% Nonlinearities
IFOGf = [Sigmoid(IFOG([Ir Fr Or], :)); tanh(IFOG(Gr, :))];

c = IFOGf(Ir, :) .* IFOGf(Gr, :) + IFOGf(Fr, :) .* c_prev;
h = IFOGf(Or, :) .* tanh(c);

end
