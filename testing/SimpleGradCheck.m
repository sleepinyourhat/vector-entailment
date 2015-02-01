% SimpleGradCheck.m

x1 = [0.5; -0.5];
x2 = [-0.5; -1];
cp = [10; 0.0001];
hp = [0.2; -0.1];
DIM = 2;
SCALE = 0.05;
WLSTM = rand(DIM * 4, DIM * 2 + 1, 1) .* (2 * SCALE) - SCALE;

[ h1, c1, IFOGf1 ] = ComputeLSTMLayer(WLSTM, hp, cp, x1);
[ h2, c2, IFOGf2 ] = ComputeLSTMLayer(WLSTM, h1, c1, x2);

[ dWLSTM_1, delta_x_down_2, delta_h_back, delta_c_back ] = ComputeLSTMLayerGradients(x2, WLSTM, IFOGf2, c1, h1, c2, [0; 0], [1; 0])
[ dWLSTM_2, delta_x_down_1, delta_h_back, delta_c_back ] = ComputeLSTMLayerGradients(x1, WLSTM, IFOGf1, cp, hp, c1, delta_h_back, delta_c_back)

dWLSTM = dWLSTM_2 + dWLSTM_1;

dWLSTMnum = 0 .* dWLSTM;

epsi = 1e-8;
for i = 1:length(dWLSTM(:))
	tempWLSTM = WLSTM;
	tempWLSTM(i) = WLSTM(i) + epsi;
	[h, c] = ComputeLSTMLayer(tempWLSTM, hp, cp, x1);
	[~, p] = ComputeLSTMLayer(tempWLSTM, h, c, x2);
	tempWLSTM(i) = WLSTM(i) - epsi;
	[h, c] = ComputeLSTMLayer(tempWLSTM, hp, cp, x1);
	[~, m] = ComputeLSTMLayer(tempWLSTM, h, c, x2);
	dWLSTMnum(i) = (p(1) - m(1)) ./ (2 * epsi);
end

scaled = abs(dWLSTMnum - dWLSTM) ./ (abs(dWLSTM) + abs(dWLSTMnum) + eps);

dWLSTMnum

dWLSTM

scaled