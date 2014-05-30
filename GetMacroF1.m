function mF1 = GetMacroF1(confusion)

P = [];
R = [];

for class = find([sum(confusion) ~= 0])
   R(class) = confusion(class, class) /  sum(confusion(:, class));
end

for class = find([sum(confusion') ~= 0])
   P(class) = confusion(class, class) /  sum(confusion(class, :));
end

avP = mean(P);
avR = mean(R);

mF1 = 2 * (avP * avR) / (avP + avR);

end