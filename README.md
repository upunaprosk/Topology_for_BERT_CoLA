# Code for the Sections 4.3/C.2 of the paper [Acceptability Judgements via Examining the Topology of Attention Maps](https://aclanthology.org/2022.findings-emnlp.7/).
## Analysis of the Feature Space

Conducted using [SHapley Additive exPlanations](https://github.com/slundberg/shap) method introduced by [Lundberg and Lee (2017)](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).  

Setting: fine-tuned En-BERT + $\textit{TDA}$ features.

- feature_space_analysis_pipeline - Pipeline declaration and hyperparameters search;
- feature_space_analysis_shapley - Principal components interpretation.
