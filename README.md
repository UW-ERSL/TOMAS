# [TOMAS: Topology Optimization of Multiscale Fluid Devices using Variational Autoencoders and Super-Shapes](https://arxiv.org/abs/2309.08435)

[Rahul Kumar Padhy](https://sites.google.com/view/rahulkp/home), [Krishnan Suresh](https://directory.engr.wisc.edu/me/faculty/suresh_krishnan),  [Aaditya Chandrasekhar](https://www.aadityacs.com/)



## Abstract

In this paper, we present a framework for multiscale topology optimization of fluid-flow devices. The objective is to minimize dissipated power, subject to a desired contact-area. The proposed strategy is to design optimal microstructures in individual finite element cells, while simultaneously optimizing the overall fluid flow. In particular, parameterized super-shape microstructures are chosen here to represent microstructures since they exhibit a wide range of permeability and contact area. To avoid repeated homogenization, a finite set of these super-shapes are analyzed a priori, and a variational autoencoder (VAE) is trained on their fluid constitutive properties (permeability), contact area and shape parameters. The resulting differentiable latent space is integrated with a coordinate neural network to carry out a global multi-scale fluid flow optimization. The latent space enables the use of new microstructures that were not present in the original data-set. The proposed method is illustrated using numerous examples in 2D.

## Citation

```

@misc{padhy2023tomastopologyoptimizationmultiscale,
      title={TOMAS: Topology Optimization of Multiscale Fluid Devices using Variational Autoencoders and Super-Shapes}, 
      author={Rahul Kumar Padhy and Krishnan Suresh and Aaditya Chandrasekhar},
      year={2023},
      eprint={2309.08435},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2309.08435}, 
}
```

