#!/bin/bash
if [ $# -lt 3 ]
  then
    echo "[ERROR]: You must specify whether you want to 0=print, 1=write,
a {seed}, if you want to 0=normal,1=shuffle,2=balance the data and lastely
which method of search you desire 0=LocalSearch 1=SimulatedAnnealing"
    exit
fi
# Get Script Directory to later find the bin path
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

if [ -z "$1" ]
  then
    echo "[ERROR]: Couldn't read the streambus value for some reason"
    exit
fi

if [ -z "$2" ]
  then
    echo "[ERROR]: Couldn't read the seed value for some reason"
    exit
fi

if [ -z "$3" ]
  then
    echo "[ERROR]: Couldn't read the shuffle value for some reason"
    exit
fi

if [ -z "$4" ]
  then
    echo "[ERROR]: Couldn't read the searchtype value for some reason"
    exit
fi


echo "[START-1]: Doing ILS search in ionosphere.arrf"
$SCRIPT_DIR/../bin/ILS ionosphere.arff b g $1 $2 $3 $4
echo "[START-2]: Doing ILS search in parkinson.arrf"
$SCRIPT_DIR/../bin/ILS parkinsons.arff 1 2 $1 $2 $3 $4
echo "[START-3]: Doing ILS search in spectf-heart.arrf"
$SCRIPT_DIR/../bin/ILS spectf-heart.arff 1 2 $1 $2 $3 $4

