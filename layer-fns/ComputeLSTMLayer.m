% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ h, c, IFOGf ] = ComputeLSTMLayer(WLSTM, h_prev, c_prev, x)
% Compute the LSTM activation.

% Based on an implementation by A. Karpathy here:
% https://github.com/karpathy/neuraltalk/blob/master/imagernn/lstm_generator.py

DIM = length(h_prev);

in = [1; x; h_prev];

IFOG = (WLSTM * in);

Ir = 0 * DIM + 1:1 * DIM;
Fr = 1 * DIM + 1:2 * DIM;
Or = 2 * DIM + 1:3 * DIM;
Gr = 3 * DIM + 1:4 * DIM;

% Nonlinearities
IFOGf([Ir Fr Or]) = Sigmoid(IFOG([Ir Fr Or]));

IFOGf(Gr) = TanhActivation(IFOG(Gr));

% Cell activation
c = IFOGf(Ir)' .* IFOGf(Gr)' + IFOGf(Fr)' .* c_prev;

h = IFOGf(Or)' .* TanhActivation(c);

end