% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ dWLSTM, delta_x_down, delta_h_back, delta_c_back ] = ComputeLSTMLayerGradients(x, WLSTM, IFOGf, c_prev, h_prev, c, delta_h, delta_c)
% Compute the LSTM activation.

% Based on an implementation by A. Karpathy here:
% https://github.com/karpathy/neuraltalk/blob/master/imagernn/lstm_generator.py

[ D, B ] = size(x);

Ir = 0 * D + 1:1 * D;
Fr = 1 * D + 1:2 * D;
Or = 2 * D + 1:3 * D;
Gr = 3 * D + 1:4 * D;

tanhC = tanh(c);
dIFOGf = IFOGf .* 0;  % Inherits gpu/cpu status from IFOGf
dIFOGf(Or, :) = (tanhC .* delta_h);

dC = delta_c + (1 - tanhC .^ 2) .* IFOGf(Or, :) .* delta_h;

if nargout > 2 % If we aren't the first node
	dIFOGf(Fr, :) = c_prev .* dC;
	delta_c_back = IFOGf(Fr, :) .* dC;
end

dIFOGf(Ir, :) = IFOGf(Gr, :) .* dC;
dIFOGf(Gr, :) = IFOGf(Ir, :) .* dC;

% Backprop through nonlinearities
dIFOG = IFOGf .* 0;  % Inherits gpu/cpu status from IFOGf
dIFOG(Gr, :) = (1 - (IFOGf(Gr, :) .^ 2)) .* dIFOGf(Gr, :);
y = IFOGf([Ir Fr Or], :);
dIFOG([Ir Fr Or], :) = (y .* (1.0 - y)) .* dIFOGf([Ir Fr Or], :);

% Compute main gradients and deltas.
dWLSTM = dIFOG * [ones(1, B); x; h_prev]';
dHin = WLSTM' * dIFOG;
delta_x_down = dHin(2:D + 1, :);
if nargout > 2
	delta_h_back = dHin(D + 2:2 * D + 1, :);
end

end
