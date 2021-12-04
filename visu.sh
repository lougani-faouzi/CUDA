#!/bin/sh

var="A000"

echo "set term x11 0" > tmp.gnu
echo "set pm3d map" >> tmp.gnu
echo "splot \"$var\"" >> tmp.gnu

var="A001"


echo "set term x11 1" >> tmp.gnu
echo "set pm3d map" >> tmp.gnu
echo "splot \"$var\"" >> tmp.gnu

echo "pause -1 \"Tapez Return pour quitter\"" >> tmp.gnu

gnuplot tmp.gnu
rm tmp.gnu
