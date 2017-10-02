# Sequential Matching Network
Keras implementation of the Sequential Matching Network model.

## data
https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip?dl=0

## Experiments

### ubuntu dialogue corpus
| Models | R@1 | R@2 | R@10 |
|:-----------:|:------------:|:------------:|:------------:|
| SMN_last* | 0.723 | 0.842 | 0.956 |
| SMN_last | 0.698 | 0.812 | 0.936 |

\* means the results reported from the original papers

## Reference
theano implementation(auther's code): https://github.com/MarkWuNLP/MultiTurnResponseSelection

paper: Wu, Yu, et al. 'Sequential Matching Network: A New Archtechture for Multi-turn Response Selection in Retrieval-based Chatbots.' ACL. 2017.
