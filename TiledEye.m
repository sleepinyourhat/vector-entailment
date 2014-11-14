function m = TiledEye(x, y)
% Creates a matrix that contains as few tiled identity matrices as are necessary to fit in the given dimensions.
% Every row and every column will have at least one non-zero.
% tiledEye(2, 5):
% 1 0 1 0 1
% 0 1 0 1 0

if x >= y
	e = eye(y);
	m = e;

	remaining = x - y;
	while remaining > y;
		m = [m; e];
		remaining = remaining - y;
	end

	% Append a partial identity.
	m = [m; e(1:remaining, :)];
else
	e = eye(x);
	m = e;

	remaining = y - x;
	while remaining > x;
		m = [m e];
		remaining = remaining - x;
	end

	% Append a partial identity.
	m = [m e(:, 1:remaining)];
end

assert(size(m, 1) == x)
assert(size(m, 2) == y)

end