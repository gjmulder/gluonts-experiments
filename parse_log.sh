#!/bin/sh

awk -F\= 'BEGIN {print "Epoch, Loss";cnt=0} /epoch_loss/ {print cnt ", " $NF;cnt+=10}' "$1" > ${1}.csv
