#! /usr/bin/env python
import sys
import os
import sferes
sys.path.insert(0, sys.path[0]+'/waf_tools')
print sys.path[0]


from waflib.Configure import conf

def build(bld):
    

    # simple_ebn
    bld.program(features = 'cxx',
                source = 'ex_sample.cpp',
                includes = '. ../../',
                uselib = 'TBB BOOST EIGEN PTHREAD MPI',
                use = 'sferes2',
                target = 'ex_sample')


