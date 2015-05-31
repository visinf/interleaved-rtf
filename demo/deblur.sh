#! /bin/bash
#
# this script requires a compiled binary of Deblur.cpp, runs from
# build/Deblur/ and assumes interleaved evaluation
#
# example call: deblur.sh input.png kernel.dlm output.png
#

filename=${1%.png}

if [ -f $1 ]
then
   cp $1 ../../demo/images/
else
   echo "$1 does not exist"
   exit 1
fi

if [ -f $2 ]
then
   cp $2 ../../demo/initial/${filename}_kernel.dlm
else
   echo "$2 does not exist"
   exit 1
fi

echo $filename > ../../demo/test.txt

./Deblur

cp ../../demo/predictions/3/${filename}_deblur.png $3
