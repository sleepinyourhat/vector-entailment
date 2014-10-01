with open("sick_data/parsed_entailment_pairs.tsv") as f:
    for i, line in enumerate(f):
        if i % 15 in [10, 11, 12, 13, 14]:
            print line
