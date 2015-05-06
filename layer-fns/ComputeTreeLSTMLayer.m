% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ h, c, IFOGf ] = ComputeTreeLSTMLayer(WLSTM, h_prev_l, h_prev_r, c_prev_l, c_prev_r)
% Compute the LSTM activation for a batch of inputs.

% Based on the binary tree model in Tai, Socher, and Manning, 2015

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

IFOG = WLSTM * in;

% Nonlinearities
IFOGf = [Sigmoid(IFOG([Ir Flr Frr Or], :)); tanh(IFOG(Gr, :))];

c = IFOGf(Ir, :) .* IFOGf(Gr, :) + IFOGf(Flr, :) .* c_prev_l + IFOGf(Frr, :) .* c_prev_r;
h = IFOGf(Or, :) .* tanh(c);

end
