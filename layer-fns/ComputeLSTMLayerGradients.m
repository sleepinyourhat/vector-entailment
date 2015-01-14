% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ dWLSTM, delta_x_down, delta_h_back, delta_c_back ] = ComputeLSTMLayerGradients(x, WLSTM, IFOGf, c_prev, h_prev, c, delta_h, delta_c)
% Compute the LSTM activation.

% Based on an implementation by A. Karpathy here:
% https://github.com/karpathy/neuraltalk/blob/master/imagernn/lstm_generator.py

DIM = length(c_prev);
N = length(x);

assert(~isempty(IFOGf));
assert(~isempty(c));
assert(~isempty(c_prev));
assert(~isempty(delta_h));
assert(~isempty(delta_c));

Ir = 0 * DIM + 1:1 * DIM;
Fr = 1 * DIM + 1:2 * DIM;
Or = 2 * DIM + 1:3 * DIM;
Gr = 3 * DIM + 1:4 * DIM;

dIFOGf = zeros(size(IFOGf, 1), size(IFOGf, 2));

dIFOGf(Or) = (c .* delta_h)';
dC = delta_c + IFOGf(Or)' .* delta_h;

if nargout > 2 % If we aren't the first node
	dIFOGf(Fr) = c_prev .* dC;
	delta_c_back = IFOGf(Fr)' .* dC;
end
dIFOGf(Ir) = IFOGf(Gr) .* dC';
dIFOGf(Gr) = IFOGf(Ir) .* dC';

% Backprop through nonlinearities
dIFOG(Gr) = TanhDeriv(IFOGf(Gr)) .* dIFOGf(Gr);
y = IFOGf([Ir Fr Or]);
dIFOG([Ir Fr Or]) = (y .* (1.0 - y)) .* dIFOGf([Ir Fr Or]);

dWLSTM = [1; x; h_prev] * dIFOG;
dHin = WLSTM' * dIFOG';
delta_x_down = dHin(2:DIM + 1);
if nargout > 2
	delta_h_back = dHin(DIM + 2:2 * DIM + 1);
end

end
