% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ dWLSTM, delta_h_l, delta_h_r, delta_c_l, delta_c_r ] = ComputeTreeLSTMLayerGradients(WLSTM, IFOGf, h_prev_l, h_prev_r, c_prev_l, c_prev_r, c, delta_h, delta_c)
% Compute the LSTM activation.

% Based on the binary tree model in Tai, Socher, and Manning, 2015

% x (D x B): The input. This should be a zero vector for non-leaf nodes.
% h_prev_* (D x B): The previous hidden states, which can be inputs from other sources.
% c_prev_* (D x B): The previous cell values. This can be zero for leaf nodes.

% WLSTM (2D + 1 x 5D): The main weight matrix
%% Inputs: [1, h_prev_l, h_prev_r]
%% Outputs: [I, Fl, Fr, O, G]

[ D, B ] = size(h_prev_l);

Ir = 0 * D + 1:1 * D;
Flr = 1 * D + 1:2 * D;
Frr = 2 * D + 1:3 * D;
Or = 3 * D + 1:4 * D;
Gr = 4 * D + 1:5 * D;

in = padarray([h_prev_l; h_prev_r], 1, 1, 'pre');

if isempty(IFOGf)
	IFOG = WLSTM * in;
	IFOGf = [Sigmoid(IFOG([Ir Flr Frr Or], :)); tanh(IFOG(Gr, :))];
end

tanhC = tanh(c);
dIFOGf = IFOGf .* 0;  % Inherits gpu/cpu status from IFOGf
dIFOGf(Or, :) = (tanhC .* delta_h);

dC = delta_c + (1 - tanhC .^ 2) .* IFOGf(Or, :) .* delta_h;

% Compute c deltas.
dIFOGf(Flr, :) = c_prev_l .* dC;
dIFOGf(Frr, :) = c_prev_r .* dC;
delta_c_l = IFOGf(Flr, :) .* dC;
delta_c_r = IFOGf(Frr, :) .* dC;

dIFOGf(Ir, :) = IFOGf(Gr, :) .* dC;
dIFOGf(Gr, :) = IFOGf(Ir, :) .* dC;

% Backprop through nonlinearities
dIFOG = IFOGf .* 0;  % Inherits gpu/cpu status from IFOGf
dIFOG(Gr, :) = (1 - (IFOGf(Gr, :) .^ 2)) .* dIFOGf(Gr, :);
y = IFOGf([Ir Flr Frr Or], :);
dIFOG([Ir Flr Frr Or], :) = (y .* (1.0 - y)) .* dIFOGf([Ir Flr Frr Or], :);

% Compute main gradients and deltas.
dWLSTM = dIFOG * in';
dHin = WLSTM' * dIFOG;

% Compute h deltas.
delta_h_l = dHin(2:D + 1, :);
delta_h_r = dHin(D + 2:2 * D + 1, :);

end
