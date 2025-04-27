# MGLearn 

Research code for our manuscript titled "Multiscale Neural Networks for Approximating Green's Functions" (Partially released). Questions? Please email Tianshi Xu.

## Disclaimer

This code is provided as-is, without any warranty of any kind. Use it at your own risk.

## Usage

Modify *.config and run notebooks. 

1. The default parameters are used for you to verify the correctness of your environment. Increase # of iterations, samples, domains, and quad points for better performance.

2. Make sure to increase mesh resolution if you want to use more subdomains. I use a lazy implementation that might fail if you have more samples than the number of quad points.

3. Note that in this version a very simple DD is provided for simplicity. We already depends on a lot of other packages. Use `METIS` for better partition.