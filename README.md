## Warp Segmentation + Vertex Refinement
This repository contains a CUDA-based multi-GPU vertex-centric graph processing framework based on Warp Segmentation and Vertex Refinement techniques.
- The options for this framework can be revealed by executing the program with no arguments.
- The vertex and edge structures and processing functions work similar to CSR-based graph processing in [CuSha](http://farkhor.github.io/CuSha/).
- The make files are configured for sm 3.5. Please adjust this option according to your device CUDA compute capability.
- You can use [this program](https://gist.github.com/farkhor/3852cbd7d29be77ae2ae) to minimize the vertex indices, without changing the graph structure. You can also use [this program](https://gist.github.com/farkhor/34aaccb593022fc9fe87) to create a PageRank-suitable input file from your edge-list.

Citing
---------------------
```shell
@inproceedings{wsvr,
 author = {Khorasani, Farzad and Gupta, Rajiv and Bhuyan, Laxmi N.},
 title = {Scalable SIMD-Efficient Graph Processing on GPUs},
 booktitle = {Proceedings of the 24th International Conference on Parallel Architectures and Compilation Techniques},
 series = {PACT '15},
 pages = {39--50},
 year = {2015}
}
```


Acknowledgements
-------------------
This work is supported by National Science Foundation grants CCF-0905509, CNS-1157377, CCF-1318103, and CCF-1524852 to UC Riverside.
