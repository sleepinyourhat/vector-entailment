DIM = 2;
SCALE = 0.05;
WLSTM = rand(DIM * 4, DIM * 2 + 1, 1) .* (2 * SCALE) - SCALE;

addpath('..')

map = containers.Map({'a', 'b'}, {1, 2});
s = Sequence.makeSequence('b', map, 1);

wordFeatures = [0.05, -0.005; -0.05, 0.01];

ETM = zeros(2, 2, 0);
ETB = zeros(2, 0);

s.updateFeatures(wordFeatures, [], WLSTM, [], ETM, ETB, @TanhActivation, 1)

hyperParams.trainWords = true;

[ dWord, ~, dWLSTM ] = ...
                   s.getGradient([1; 0], [], wordFeatures, [], ...
                        WLSTM, [], ETM, ETB, ...
                        @TanhDeriv, hyperParams);

dWordNum = 0 .* dWord;

epsi = 1e-8;
for i = 1:length(dWord(:))
	tempWord = wordFeatures;
	tempWord(i) = wordFeatures(i) + epsi;
	s.updateFeatures(tempWord, [], WLSTM, [], ETM, ETB, @TanhActivation, 1)
	p = s.getFeatures();

	tempWord(i) = wordFeatures(i) - epsi;
	s.updateFeatures(tempWord, [], WLSTM, [], ETM, ETB, @TanhActivation, 1)
	m = s.getFeatures;

	dWordNum(i) = ((p(1) - m(1)) ./ (2 * epsi));
end

dWLSTMnum = 0 .* dWLSTM;

for i = 1:length(dWLSTM(:))
	tempWLSTM = WLSTM;
	tempWLSTM(i) = WLSTM(i) + epsi;
	s.updateFeatures(wordFeatures, [], tempWLSTM, [], ETM, ETB, @TanhActivation, 1)
	p = s.getFeatures();

	tempWLSTM(i) = WLSTM(i) - epsi;
	s.updateFeatures(wordFeatures, [], tempWLSTM, [], ETM, ETB, @TanhActivation, 1)
	m = s.getFeatures;

	dWLSTMnum(i) = ((p(1) - m(1)) ./ (2 * epsi));
end

dWLSTMnum

dWLSTM

dWordNum

dWord

scaledWLSTM = abs(dWLSTMnum - dWLSTM) ./ (abs(dWLSTM) + abs(dWLSTMnum) + eps)

scaledWord = abs(dWordNum - dWord) ./ (abs(dWord) + abs(dWordNum) + eps)

