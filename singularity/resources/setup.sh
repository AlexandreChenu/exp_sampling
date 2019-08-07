#!/bin/bash

#./waf configure --exp exp_sampling --dart /workspace --kdtree /workspace/include --robot_dart /workspace
./waf configure --kdtree /workspace/include
./waf --exp exp_sampling
