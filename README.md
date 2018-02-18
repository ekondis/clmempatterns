# clmempatterns
This benchmark serves as a tool for investigating memory bandwidth of GPU devices by fully loading a linear memory space by applying different access strides. It is implemented using OpenCL API and therefore it can be used as a cross vendor tool. The size of memory space, the elements used and the granularity can be configured. All configuration parameters are set via macros on kernel source code level in order to be effectively taken into account on compilation time.

Usage
-----

Usage: `clmempatterns [options] {device index} [index magnitude [grid magnitude [workgroup magnitude [vector width]]]]`

Use '-h' parameter to view all valid options.

Configuration parameters
--------------

Configuration consists of the following parameters:

1. Index magnitude

 This defines the index space size as a power of 2, i.e. by using X as value, a total amount of 2<sup>X</sup> elements will be set for the benchmark. Default setting is 22, i.e. 2<sup>22</sup>=4194304 total elements.

2. Grid magnitude

 This parameter defines the size of the grid as a power of 2, i.e. a value of X sets an NDRange of 2<sup>X</sup> total workitems. Default setting is 18, i.e. 2<sup>18</sup>=262144 total workitems.

 Index and grid magnitudes define granularity as: *Granularity*=*IndexSpaceSize*/*GridSize*

3. Workgroup magnitude

 This parameter defines the workgroup size, again as a power of 2. Default setting is 8, i.e. 2<sup>8</sup>=256 workitems per workgroup.

4. Vector width

 Can be 1 (scalar), 2, 4, or 8. Default setting is 2 (type int2).

Building program
--------------

Make sure that OCLSDKDIR (plus OCLSDKINC & OCLSDKLIB) is set correctly in "Makefile" (if required) and then proceed to make:

```
make
```

Example executions
--------------

Running the benchmark without arguments shows the valid options and the available OpenCL devices that you may experiment with. You may execute the benchmark by just setting the OpenCL device index as `./clmempatterns 1`.

For more controlled benchmark settings you may set all parameters. For instance, on AMD/ROCm platform an example execution is as follows:

```
$ ./clmempatterns 1 26 20 8 1
clmempatterns rel. 0.3git
Developed by Elias Konstantinidis (ekondis@gmail.com)

Selected platform: AMD Accelerated Parallel Processing
Selected device  : gfx803
Driver version   : 2545.0 (HSA1.1,LC)
Address bits     : 64

Benchmark parameters
index space     : 67108864
vector width    : 1 (type: int )
required memory : 256 MB
grid space      : 1048576 (4096 workgroups)
workgroup size  : 256
granularity     : 64
allocated buffer: device

Platform initialization
Creating buffer...   Ok
Building programs... [#####################]
Creating kernels...  Ok
Zeroing buffer...    Ok

Experimental execution
Running... [#####################]

Summary:
Stride magnitude  0: Bandwidth  17.777 GB/sec (avg time  15.100354 msecs) *Serial accesses
Stride magnitude  1: Bandwidth  46.266 GB/sec (avg time   5.802001 msecs)
Stride magnitude  2: Bandwidth  70.492 GB/sec (avg time   3.808029 msecs)
Stride magnitude  3: Bandwidth 132.325 GB/sec (avg time   2.028613 msecs)
Stride magnitude  4: Bandwidth 157.876 GB/sec (avg time   1.700297 msecs)
Stride magnitude  5: Bandwidth 332.622 GB/sec (avg time   0.807029 msecs)
Stride magnitude  6: Bandwidth 374.789 GB/sec (avg time   0.716231 msecs)
Stride magnitude  7: Bandwidth 379.795 GB/sec (avg time   0.706790 msecs)
Stride magnitude  8: Bandwidth 370.526 GB/sec (avg time   0.724471 msecs) *Workgroup striding
Stride magnitude  9: Bandwidth 436.231 GB/sec (avg time   0.615352 msecs)
Stride magnitude 10: Bandwidth 449.315 GB/sec (avg time   0.597432 msecs)
Stride magnitude 11: Bandwidth 474.542 GB/sec (avg time   0.565673 msecs)
Stride magnitude 12: Bandwidth 483.988 GB/sec (avg time   0.554633 msecs)
Stride magnitude 13: Bandwidth 489.281 GB/sec (avg time   0.548633 msecs)
Stride magnitude 14: Bandwidth 494.472 GB/sec (avg time   0.542873 msecs)
Stride magnitude 15: Bandwidth 493.526 GB/sec (avg time   0.543913 msecs)
Stride magnitude 16: Bandwidth 492.802 GB/sec (avg time   0.544713 msecs)
Stride magnitude 17: Bandwidth 491.863 GB/sec (avg time   0.545753 msecs)
Stride magnitude 18: Bandwidth 490.497 GB/sec (avg time   0.547273 msecs)
Stride magnitude 19: Bandwidth 490.425 GB/sec (avg time   0.547353 msecs)
Stride magnitude 20: Bandwidth 442.735 GB/sec (avg time   0.606312 msecs) *Grid striding
```

Here is a chart (gnuplot generated) illustrating the above execution data:
![R9-Nano ROCm execution results](https://raw.githubusercontent.com/ekondis/clmempatterns/other-stuff/img/fiji.png "clmempatterns example execution")
