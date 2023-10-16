HIP Transfer Streams (HIts)
============================

This application is designed to launch intra-node transfer streams in an
adjustable way. It may trigger different types of transfers concurrently.
Each transfer is bound to a stream. Transfer buffers in main memory are
allocated (by default) on the proper NUMA node.


Dependencies
------------

* **Rocm**
* **libnuma**


How to build HIts
-----------------

    % make


How to run HIts
---------------

    % ./hits [ARGS...]

Arguments are :

    -d, --dtoh=<id>            Provide GPU id for Device to Host transfer.
    -h, --htod=<id>            Provide GPU id for Host to Device transfer.
    -i, --iter=<nb>            Specify the amount of iterations. [default: 100]
    -n, --no-numa-affinity     Do not make the transfer buffers NUMA aware.
    -p, --dtod=<id,id>         Provide comma-separated GPU ids to specify which
                               pair of GPUs to use for peer to peer transfer.
                               First id is the destination, second id is the
                               source.
    -s, --size=<bytes>         Specify the transfer size in bytes. [default:
                               1073741824]
    -?, --help                 Give this help list
        --usage                Give a short usage message
    -V, --version              Print program version


CUDA Version
-----------

To run on NVIDIA GPUs, check the CUDA version [CUts](https://github.com/jyvet/cuts).
