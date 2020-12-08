# Awesome Vision and Language PreTrain Models (PTMs)
Maintained by [Yang Gao]() (ustcgaoy01@gmail.com) Last Update on 12/05/2020. 

Due to the large amount of research in this field, we mainly focus on Vision-Only PTMs, Language-Only PTMs, Multimodal PTMs, and other releated research in this field such as transfer learning. 

## Table of Contents
* [Surveys](#survey)
* [Transformers](#transformers)
* [Vision-Only PTMs](#vision-only-ptms)
* [Language-Only PTMs](#language-only-ptms)
* [MultiModal/Vision-Language PTMs](#multimodal-ptms)

## Survey
[Efficient Transformers: A Survey](https://arxiv.org/pdf/2009.06732.pdf)

[Transformers: State-of-the-Art Natural Language Processing](https://www.aclweb.org/anthology/2020.emnlp-demos.6.pdf)

[O(n) Connections are Expressive Enough: Universal Approximability of Sparse Transformers](https://proceedings.neurips.cc/paper/2020/file/9ed27554c893b5bad850a422c3538c15-Paper.pdf), NIPS, 2020.

## Transformers
![](https://github.com/AlenUbuntu/Awesome-Vision-and-Language-PreTrain-Models/blob/main/transformers.png)

### Efficiency and Performance
**Performer**

[Masked language modeling for proteins via linearly scalable long-context transformers](https://arxiv.org/pdf/2006.03555.pdf), arXiv, 2020.

**Linformer**

[Linformer: Selfattention with linear complexity.](https://arxiv.org/pdf/2006.04768.pdf), arXiv, 2020.

**Linear Transformers**

[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/pdf/2006.16236.pdf), arXiv, 2020.

**BigBird**

[Big bird: Transformers for longer sequences.](https://arxiv.org/pdf/2007.14062.pdf), arXiv, 2020.

**Synthesizer**

[Synthesizer: Rethinking self-attention in transformer models.](https://arxiv.org/pdf/2005.00743.pdf), arXiv, 2020.

**ETC**

[Etc: Encoding long and structured data in transformers.](https://arxiv.org/pdf/2004.08483.pdf), arXiv, 2020.

**Longformer**

[Longformer: The long-document transformer.](https://arxiv.org/pdf/2004.05150.pdf), arXiv, 2020.

**Sinkhorn Transformer**

[Sparse sinkhorn attention.](https://arxiv.org/pdf/2002.11296.pdf), arXiv, 2020.

**Compressive Transformer**

[Compressive transformers for long-range sequence modelling.](https://arxiv.org/pdf/1911.05507.pdf), ICLR, 2020.

**Routing Transformer**

[Efficient Content-Based Sparse Attention with Routing Transformers](https://arxiv.org/pdf/2003.05997.pdf), arXiv, 2020.

**Reformer**

[Reformer: The efficient transformer.](https://arxiv.org/pdf/2001.04451.pdf), ICLR, 2020.

**FPT**

[Feature Pyramid Transformer](https://arxiv.org/pdf/2007.09451.pdf), ECCV, 2020.

**Sandwitch Transformer**

[Improving Transformer Models by Reordering their Sublayers](https://www.aclweb.org/anthology/2020.acl-main.270.pdf), ACL, 2020.

**Highway Transformer**

[Highway Transformer: Self-Gating Enhanced Self-Attentive Networks](https://www.aclweb.org/anthology/2020.acl-main.616.pdf), ACL, 2020.

**Cascade Transformer**

[The Cascade Transformer: an Application for Efficient Answer Sentence Selection](https://www.aclweb.org/anthology/2020.acl-main.504.pdf), ACL, 2020.

**Hard-Aware Transformer**

[HAT: Hardware-Aware Transformers for Efficient Natural Language Processing](https://www.aclweb.org/anthology/2020.acl-main.686.pdf), ACL, 2020.

**Memory-driven Transformer**

[Generating Radiology Reports via Memory-driven Transformer](https://www.aclweb.org/anthology/2020.emnlp-main.112.pdf), EMNLP, 2020.

**Funnel-Transformer**

[Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing](https://proceedings.neurips.cc/paper/2020/file/2cd2915e69546904e4e5d4a2ac9e1652-Paper.pdf), NIPS, 2020.

**LL**

[Deep Transformers with Latent Depth](https://proceedings.neurips.cc/paper/2020/file/1325cdae3b6f0f91a1b629307bf2d498-Paper.pdf), NIPS, 2020.

**Fast Transformers with Clustered Attention**

[Fast Transformers with Clustered Attention](https://proceedings.neurips.cc/paper/2020/file/f6a8dd1c954c8506aadc764cc32b895e-Paper.pdf), NIPS, 2020.

**PLD**

[Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping](https://proceedings.neurips.cc/paper/2020/file/a1140a3d0df1c81e24ae954d935e8926-Paper.pdf), NIPS, 2020.

**Axial Transformer**

[Axial attention in multidimensional transformers.](https://arxiv.org/pdf/1912.12180.pdf), arXiv, 2019.

**Sparse Transformer**

[Generating long sequences with sparse transformers.](https://arxiv.org/pdf/1904.10509.pdf), arXiv, 2019.

**Transformer-XL**

[Transformer-xl: Attentive language models beyond a fixed-length context.](https://arxiv.org/pdf/1901.02860.pdf), ACL, 2019.

**Adaptive-Span**

[Adaptive Attention Span in Transformers](https://www.aclweb.org/anthology/P19-1032.pdf), ACL, 2019.

**Set Transformer**

[Set transformer: A framework for attention-based permutation-invariant neural networks.](https://arxiv.org/pdf/1810.00825.pdf), ICML, 2019.

**Levenshtein Transformer**

[Levenshtein Transformer](https://proceedings.neurips.cc/paper/2019/file/675f9820626f5bc0afb47b57890b466e-Paper.pdf), NIPS, 2019.

**Memory Compressed**

[Generating wikipedia by summarizing long sequences](https://arxiv.org/pdf/1801.10198.pdf), ICLR, 2018.

**Transformer**

[Attention is all you need.](https://arxiv.org/pdf/1706.03762.pdf), NuerIPS, 2017

### Vision
**Epipolar Transformers**

[Epipolar Transformers](https://arxiv.org/pdf/2005.04551.pdf), CVPR, 2020.

**Texture Transformer**

[Learning Texture Transformer Network for Image Super-Resolution](https://arxiv.org/pdf/2006.04139.pdf), CVPR, 2020.

**SE(3)-Transformers**

[SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks](https://proceedings.neurips.cc/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf), NIPS, 2020.

**Image Transformer**

[Image transformer.](https://arxiv.org/pdf/1802.05751.pdf), ICML, 2018.

### Transfer Learning
**Style Transformer**

[Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation](https://www.aclweb.org/anthology/P19-1601.pdf), ACL, 2019.

### MultiModal
**Sign Language Transformers**

[Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://arxiv.org/pdf/2003.13830.pdf), CVPR, 2020.

**Multimodal Transformer**

[Multimodal Transformer for Multimodal Machine Translation.](https://www.aclweb.org/anthology/2020.acl-main.400.pdf), ACL, 2020.

**Meshed-Memory Transformer**

[Meshed-Memory Transformer for Image Captioning](https://arxiv.org/pdf/1912.08226.pdf), CVPR, 2020.

**MART**

[MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning](https://www.aclweb.org/anthology/2020.acl-main.233.pdf), ACL, 2020.

**MulT**

[Multimodal Transformer for Unaligned Multimodal Language Sequences](https://www.aclweb.org/anthology/P19-1656.pdf), ACL, 2019.

### Graph Transformer
**HetGT**

[Heterogeneous Graph Transformer for Graph-to-Sequence Learning](https://www.aclweb.org/anthology/2020.acl-main.640.pdf), ACL, 2020.

**GTN**

[Graph Transformer Networks](https://proceedings.neurips.cc/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf), NIPS, 2019.

## Vision-Only PTMs
### Well-Known Pretrain Models

**Vision Transformer**

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf), ICLR, 2021.

**LambdaNetworks**

[LambdaNetworks: Modeling long-range Interactions without Attention](https://openreview.net/pdf?id=xTJEN-ggl1b), ICLR, 2021.

**IPT**

[Pre-Trained Image Processing Transformer](https://arxiv.org/pdf/2012.00364.pdf), arXiv, 2020.

**DETR**

[End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf), arXiv, 2020.

**DEFORMABLE DETR**

[Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/pdf/2010.04159.pdf), arXiv, 2020.

**Epipolar Transformers**

[Epipolar Transformers](https://arxiv.org/pdf/2005.04551.pdf), CVPR, 2020.

**Sketchformer**

[Sketchformer: Transformer-based Representation for Sketched Structure](https://arxiv.org/pdf/2002.10381.pdf), CVPR, 2020.

**Texture Transformer Network**

[Learning Texture Transformer Network for Image Super-Resolution](https://arxiv.org/pdf/2006.04139.pdf), CVPR, 2020.

**iGPT**

[Generative Pretraining from Pixels](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf), ICML, 2020.

**FPT**

[Feature Pyramid Transformer](https://arxiv.org/pdf/2007.09451.pdf), ECCV, 2020.

**RelationNet++**

[RelationNet++: Bridging Visual Representations for Object Detection via Transformer Decoder](https://proceedings.neurips.cc/paper/2020/file/9d684c589d67031a627ad33d59db65e5-Paper.pdf), NIPS, 2020.

**SE(3)-Transformers**

[SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks](https://proceedings.neurips.cc/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf), NIPS, 2020.

### Other Topics
[Modeling Techniques, Transfer Learning and Applications](https://github.com/AlenUbuntu/Awesome-Vision-and-Language-PreTrain-Models/blob/main/VisionOnlyPTMs.md)

## Language-Only PTMs
### Well-Known Pretrain Models
**ELECTRA** 

[ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS](https://arxiv.org/pdf/2003.10555.pdf), ICLR, 2020.

**ALBERT** 

[ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS](https://arxiv.org/pdf/1909.11942.pdf), ICLR, 2020.

**MINILM**

[MINILM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://proceedings.neurips.cc/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), NIPS, 2020.

**Longformer** 

[Longformer: The long-document transformer.](https://arxiv.org/pdf/2004.05150.pdf), arXiv, 2020.

**XLM**

[Cross-lingual Language Model Pretraining](https://arxiv.org/pdf/1901.07291.pdf), NeurIPS, 2019

**DistilBERT**

[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108.pdf), NeurIPS, 2019

**T5**

[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf), JMLR, 2019.

**Bart**

[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf), ACL, 2019.

**XLNet**

[XLNet: Generalized Autoregressive Pretraining for Language Understanding.](https://proceedings.neurips.cc/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf), NIPS, 2019.

**Transformer-XL**

[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf), ACL, 2019.

**GPT/GPT2**

[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), OpenAI blog, 2019

**RoBERTa**

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf), arXiv, 2019.

**Bert**

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf), NAACL, 2019.

**Ouroboros**

[Ouroboros: On Accelerating Training of Transformer-Based Language Models](https://proceedings.neurips.cc/paper/2019/file/1b79b52d1bf6f71b2b1eb7ca08ed0776-Paper.pdf), NIPS, 2019.

### Other Topics
[Modeling Techniques, Transfer Learning and Applications](https://github.com/AlenUbuntu/Awesome-Vision-and-Language-PreTrain-Models/blob/main/LanguageOnlyPTMs.md)

## MultiModal PTMs
### Well-Known Pretrain Models
**MMBT**

[Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/pdf/1909.02950.pdf), arXiv, 2019

**Sign Language Transformers**

[Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://arxiv.org/pdf/2003.13830.pdf), CVPR, 2020.

**Meshed-Memory Transformer**

[Meshed-Memory Transformer for Image Captioning](https://arxiv.org/pdf/1912.08226.pdf), CVPR, 2020.

**SCT**

[SCT: Set Constrained Temporal Transformer for Set Supervised Action Segmentation](https://arxiv.org/pdf/2003.14266.pdf), CVPR, 2020.

**M4C**
[Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA](https://arxiv.org/pdf/1911.06258.pdf), CVPR, 2020.

**MAG-BERT, MAG-XLNet**

[Integrating Multimodal Information in Large Pretrained Transformers.](https://www.aclweb.org/anthology/2020.acl-main.214.pdf), ACL, 2020.

**MART**

[MART: Memory-Augmented Recurrent Transformer for Coherent Video Paragraph Captioning](https://www.aclweb.org/anthology/2020.acl-main.233.pdf), ACL, 2020.

**TaBERT**

[TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data.](https://www.aclweb.org/anthology/2020.acl-main.745.pdf), ACL, 2020.

**AV-ASR**

[Multiresolution and Multimodal Speech Recognition with Transformers](https://www.aclweb.org/anthology/2020.acl-main.216.pdf), ACL, 2020.

**Multimodal Transformer**

[Multimodal Transformer for Multimodal Machine Translation.](https://www.aclweb.org/anthology/2020.acl-main.400.pdf), ACL, 2020.

**Unified Multimodal Transformer**

[Improving Multimodal Named Entity Recognition via Entity Span Detection with Unified Multimodal Transformer](https://www.aclweb.org/anthology/2020.acl-main.306.pdf), ACL, 2020.

**VGD-GPT2**

[Video-Grounded Dialogues with Pretrained Generation Language Models.](https://www.aclweb.org/anthology/2020.acl-main.518.pdf), ACL, 2020.

**X-LXMERT**

[X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers](https://www.aclweb.org/anthology/2020.emnlp-main.707.pdf), EMNLP, 2020.

**COOT**

[COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning](https://proceedings.neurips.cc/paper/2020/file/ff0abbcc0227c9124a804b084d161a2d-Paper.pdf), NIPS, 2020.

**MTN**

[Multimodal Transformer Networks for End-to-End Video-Grounded Dialogue Systems](https://www.aclweb.org/anthology/P19-1564.pdf), ACL, 2019.

**MulT**

[Multimodal Transformer for Unaligned Multimodal Language Sequences](https://www.aclweb.org/anthology/P19-1656.pdf), ACL, 2019.

### Special Topic
[Vision-Language-PTMs](https://github.com/AlenUbuntu/Awesome-Vision-and-Language-PreTrain-Models/blob/main/VL-PTMs.md)

