set terminal postscript color enhanced
set output "part_en_vs_t.eps"
set xlabel "t [s]"
set ylabel "E [J]"

plot "particles_energy.dat" u 1:5 with lines lt rgb "blue" title "Part. #4",\
"" u 1:10 with lines lt rgb "red" title "Part. #9",\
"" u 1:20 with lines lt rgb "orange" title "Part. #19"
