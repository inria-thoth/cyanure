all:
	latex figure.tex
	dvipdf figure.dvi
	pdflatex --shell-escape main.tex
	bibtex main
	pdflatex --shell-escape main.tex
	pdflatex --shell-escape main.tex
