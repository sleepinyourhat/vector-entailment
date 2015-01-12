% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ dWLSTM, delta_x_down, delta_h_back, delta_c_back ] = ComputeLSTMLayerGradients(x, WLSTM, IFOGf, h_prev, c_prev, c, delta_h, delta_c )
% Compute the LSTM activation.

% Based on an implementation by A. Karpathy here:
% https://github.com/karpathy/neuraltalk/blob/master/imagernn/lstm_generator.py

DIM = length(h_prev);
N = length(x);

dIFOGf = zeros(size(IFOGf, 1), size(IFOGf, 2));
dIFOGf(2 * DIM + 1:3 * DIM) = c .* delta_h;
dC = delta_c + IGOFf(2 * DIM + 1:3 * DIM) .* delta_h;

if nargout > 2 % If we aren't the first node
	dIFOGf(DIM + 1:2 * DIM) = c_prev .* dC;
	delta_c_back = IFOGf(3 * DIM + 1:4 * DIM) .* dC;
end
dIFOGf(1:DIM) = IFOGf(3 * DIM + 1:4 * DIM) .* dC;
dIFOGf(3 * DIM + 1:4 * DIM) = IFOGf(1:DIM) .* dC;

% Backprop through nonlinearities
dIFOG(3 * DIM + 1:4 * DIM) = (1 - IFOGf(3 * DIM + 1:4 * DIM) ^ 2) .* dIFOGf(3 * DIM + 1:4 * DIM);
y = IFOGf(1:3 * DIM);
dIFOG(1:3 * DIM) = (y .* (1.0 - y)) .* dIFOGf(1:3 * DIM);

dWLSTM = x * dIFOGf';
dHin = dIFOG * WLSTM';
delta_x_down = dHin(2:DIM + 1);
if nargout > 2
	delta_h_back = dHin(DIM + 2:2 * DIM + 1)
end

end