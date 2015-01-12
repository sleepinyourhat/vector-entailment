% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ h, c, IFOG, IFOGf ] = ComputeLSTMLayer(WLSTM, h_prev, c_prev, x)
% Compute the LSTM activation.

% Based on an implementation by A. Karpathy here:
% https://github.com/karpathy/neuraltalk/blob/master/imagernn/lstm_generator.py

DIM = length(h_prev);

in = [1; x; h_prev];
IFOG = in * WLSTM;

% Nonlinearities
IFOGf(1:3 * DIM) = Sigmoid(IFOG(1:3 * DIM));
IFOGf(3*DIM + 1:4*DIM) = TanhActivation(IFOG(3*DIM + 1:4*DIM));

% Cell activation
c = IFOGf(1:DIM) .* IFOGf(3 * DIM + 1:4 * DIM) + IFOGf(DIM + 1:2 * DIM) .* c_prev

h = IFOGf(2 * DIM + 1:3 * DIM) .* c;

end