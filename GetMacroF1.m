function mF1 = GetMacroF1(confusion)
% Compute the macroaveraged F1 from a confusion matrix.
% NOTE: There seems to be some disagreement in the literature
% on how to handle empty classes in this calculation.

P = [];
R = [];

for class = find([sum(confusion) ~= 0])
   R = [R, confusion(class, class) /  sum(confusion(:, class))];
   if sum(confusion(class, :)) ~= 0
    P = [P, confusion(class, class) /  sum(confusion(class, :))];
   else
    P = [P, 0];
   end
end

avP = mean(P);
avR = mean(R);

mF1 = 2 * (avP * avR) / (avP + avR);

end