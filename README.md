# A Scalable Linear-Time Algorithm for Horizontal Visibility Graph Construction Over Long Sequences


Implementation of worst-case O(n) online scalable algorithms for constructing HVGs.

Includes code to run numerical benchmarks to compare the proposed method of _dual tree horizontal visibility_ with two previous approaches: the binary search tree method [1] and the divide and conquer method [2].

[Paper proposing the method.](paper.pdf)

[Code implementing dual tree horizontal visibility graphs (DTHVGs).](dt_hvg.py)

### References

[1] Fano Yela, D., Thalmann, F., Nicosia, V., Stowell, D., Sandler, M., 2020. Online visibility graphs: Encoding visibility in a binary search tree. Phys. Rev. Research 2, 023069. https://doi.org/10.1103/PhysRevResearch.2.023069

[2] Lan, X., Mo, H., Chen, S., Liu, Q., Deng, Y., 2015. Fast transformation from time series to visibility graphs. Chaos 25, 083105. https://doi.org/10.1063/1.4927835
