set terminal postscript enhanced color 
set output "scatter_volume.eps"
set key on
set yrange[-1:4]
set xrange[0:55000]
set xtics 0,5000,55000
set title "Cantidad de puntos vs. Error relativo porcentual"
set xlabel "Puntos aleatorios totales"
set ylabel "Error relativo porcentual"
f(x,y) = (y- pi*2)/sqrt(x) 
plot "cilinder_surface_faces_volume_200.dat" u 1:4 with dots lw 6 lt rgb "red" title "200 caras",\
"cilinder_surface_faces_volume_150.dat" u 1:4 with points lw 2 lt rgb "blue" title "150 caras",\
"cilinder_surface_faces_volume_100.dat" u 1:4 with dots lw 8 lt rgb "violet" title "100 caras",\
"cilinder_surface_faces_volume_50.dat" u 1:4 with points lw 2 lt rgb "orange" title "50 caras",\
"cilinder_surface_faces_volume_20.dat" u 1:4 with dots lw 6 lt rgb "black" title "20 caras"

