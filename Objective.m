function o = Objective(trueRelation, relationProbs, params, hyperParams)

if nargin < 3
    regTerm = 0;
else 
    regTerm = (hyperParams.lambda / 2) * sum(params .^ 2);
end

o = -log(relationProbs(trueRelation)) + regTerm;

end