#!/usr/bin/env bash
# Run slam benchmark
for i in `seq 1 10`; do
    for bagfile in ./*.bag; do
        killall -9 roscore
        timeout 5m roslaunch bruce_slam slam.launch file:=`pwd`/$bagfile kill:=true
        killall -9 roscore
        mv ~/.ros/*.npz .
        sleep 3
    done
done