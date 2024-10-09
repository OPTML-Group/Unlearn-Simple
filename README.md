<div align='center'>
 
# Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning

![preprint](https://img.shields.io/badge/arXiv-TBA-B31B1B)
[![issues](https://img.shields.io/badge/Issues-Welcome!-yellow)](https://github.com/OPTML-Group/Unlearn-Saliency/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/Unlearn-Simple?tab=MIT-1-ov-file)
[![GitHub top language](https://img.shields.io/github/languages/top/OPTML-Group/Unlearn-Simple)](https://github.com/OPTML-Group/Unlearn-Simple)
[![GitHub repo size](https://img.shields.io/github/repo-size/OPTML-Group/Unlearn-Simple)](https://github.com/OPTML-Group/Unlearn-Simple)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/Unlearn-Simple)](https://github.com/OPTML-Group/Unlearn-Simple)

</div>

This is the official code repository for the paper Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning.

## Abstract

In this work, we address the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences and associated model capabilities (*e.g.*, copyrighted data or harmful content generation) while preserving essential model utilities, without the need for retraining from scratch. Despite the growing need for LLM unlearning, a principled optimization framework remains lacking. To this end, we revisit the state-of-the-art approach, negative preference optimization (NPO), and identify the issue of reference model bias, which could undermine NPO's effectiveness, particularly when unlearning forget data of varying difficulty. Given that, we propose a simple yet effective unlearning optimization framework, called **SimNPO**, showing that 'simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We also provide deeper insights into SimNPO's advantages, supported by analysis using mixtures of Markov chains. Furthermore, we present extensive experiments validating SimNPO's superiority over existing unlearning baselines in benchmarks like TOFU and MUSE, and robustness against relearning attacks.


<table align="center">
  <tr>
    <td align="center"> 
      <img src="Images/teaser.png" alt="Teaser" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> Systematic overview and experiment highlights of SimNPO.</em>
    </td>
  </tr>
</table>

## Getting Started
* [SimNPO on TOFU and WMDP](TOFU)
* [SimNPO on MUSE](MUSE)
* [SimNPO on Synthetic data](synthetic)

## Contributors

* [Chongyu Fan](https://a-f1.github.io/)
* [Jiancheng Liu](https://ljcc0930.github.io/)
* [Licong Lin](https://licong-lin.github.io/)

<!--## Cite This Work
```
```--!>
