function o = Objective(trueRelation, relationProbs)

o = -log(relationProbs(trueRelation));

end