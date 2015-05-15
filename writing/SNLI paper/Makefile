snli_paper.pdf: $(wildcard *.tex) MLSemantics.bib Makefile
	@pdflatex snli_paper
	@bibtex snli_paper
	@pdflatex snli_paper
	@pdflatex snli_paper

clean:
	rm -f *.aux *.log *.bbl *.blg snli_paper.pdf *.bak *.ps *.dvi

fast:
	@pdflatex snli_paper

default: snli_paper.pdf
