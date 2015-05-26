snli_paper.pdf: $(wildcard *.tex) MLSemantics.bib Makefile learning_curves_bow.pdf
	@pdflatex snli_paper
	@bibtex snli_paper
	@pdflatex snli_paper
	@pdflatex snli_paper

learning_curves_bow.pdf: learning_curve_lexicalized.dat learning_curve_unlexicalized.dat mk_learning_curves.plot mk_learning_curves.py
	python mk_learning_curves.py

clean:
	rm -f *.aux *.log *.bbl *.blg snli_paper.pdf *.bak *.ps *.dvi

fast:
	@pdflatex snli_paper

default: snli_paper.pdf
