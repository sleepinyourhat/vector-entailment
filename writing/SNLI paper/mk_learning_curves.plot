set terminal pdf dashed

set xlabel "# Training Examples"
set ylabel "3-Class Accuracy"
set xrange [1:600000]
set yrange [33.333:100]
#set logscale x 10

set output 'learning_curves_bow.pdf'
plot \
  "learning_curve_lexicalized.dat"   using 1:2 title 'Lexicalized'   with linespoints lt 1 pt 0 linecolor rgb "blue", \
  "learning_curve_unlexicalized.dat" using 1:2 title 'Unlexicalized' with linespoints lt 3 pt 0 linecolor rgb "black";
