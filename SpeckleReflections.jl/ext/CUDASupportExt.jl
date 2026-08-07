module CUDASupportExt
using CUDA 
using SpeckleReflections

SpeckleReflections.opt_collect(a::CuArray) = a
end