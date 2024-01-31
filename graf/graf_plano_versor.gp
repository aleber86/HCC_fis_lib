set terminal qt
set xrange[-3:3]
set yrange[-3:3]
set zrange[-3:3]

set arrow from -1,0,0 to 1,0,0
set arrow from 1,0,0 to 1,1,0
set arrow from 1,1,0 to -1,1,0
set arrow from -1,1,0 to -1,0,0
set arrow from 0,0.5,0 to 0,0.5,1

splot 0:0:0
   
