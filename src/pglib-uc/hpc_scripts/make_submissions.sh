#!/bin/bash
prefix=$1

q1="0.5"
q2="0.05,0.5,0.95,0.99,0.999"
q3="0.002,0.1,0.10,0.3,0.5,0.6,0.75,0.9,0.95,0.99,0.993,0.9967,0.999"

for num_gen in 10 20 30 40 50 60;
  do n=1 && for quantiles in $q1 $q2 $q3 ;  
    do qsub submit.sh ${prefix}_convex_g${num_gen}_q${n} $num_gen $quantiles && n=$(($n+1)) && echo ${prefix}_convex_g${num_gen}_q${n} && echo $quantiles ;
  done ;
done
