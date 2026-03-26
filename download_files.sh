#!/usr/bin/env bash

cd src/svd3x3-sycl/
wget https://raw.githubusercontent.com/kuiwuchn/3x3_SVD_CUDA/refs/heads/master/svd3x3/svd3x3/Dataset_1M.txt

cd ../

cd cc-cuda
wget https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/delaunay_n24.egr

cd ../

cd mriQ-cuda/
wget https://www.cs.ucr.edu/~nael/217-f19/labs/mri-q.tgz
tar -xvzf mri-q.tgz

cd ../

cd floydwarshall2-cuda/
wget https://userweb.cs.txstate.edu/~burtscher/research/ECL-APSP/CollegeMsg.egr

cd ../

cd sad-cuda/
tar -xvzf data.tar.gz

cd ../

cd seam-carving-cuda/
tar -xvzf image.tar.gz

cd ../
cd testSNAP-omp
tar -xvzf refdata.tar.gz 

cd ../
cd minimap2-sycl
tar -xvzf in-1k.txt.tar.gz
cd ../
