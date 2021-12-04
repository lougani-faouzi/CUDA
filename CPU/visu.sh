#!/bin/sh

var="A000"

echo "set term png" > tmp.gnu
echo "set pm3d map" >> tmp.gnu
echo "splot \"$var\"" >> tmp.gnu

var="A001"

echo "set term png" > tmp2.gnu
echo "set pm3d map" >> tmp2.gnu
echo "splot \"$var\"" >> tmp2.gnu

# echo "pause -1 \"Tapez Return pour quitter\"" >> tmp.gnu

gnuplot tmp.gnu >& convolution.png
gnuplot tmp2.gnu >& convolution2.png
rm tmp.gnu
rm tmp2.gnu
