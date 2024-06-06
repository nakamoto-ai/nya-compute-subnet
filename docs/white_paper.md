# Decentralized AI Training

## Technical Specifications <a name="technical" />

Training models with billions or trillions of parameters is daunting, and doing so in a decentralized network is even more challenging. Common distributed training methods, such as data parallelism and model parallelism, are unsuitable for decentralized networks due to the need for a central coordinator to synchronize the training process by aggregating gradients from all participants. For a 7 billion parameter model, the gradients would amount to 28 GB (7 x 10^9 parameters x 4 bytes per parameter) per node, which would need to be transferred to all other nodes. This data transfer creates a bottleneck, even in a centralized setting.

Currently, [pipeline parallelism and memory usage optimizations][zero] are the primary methods for training neural networks at that scale. Unfortunately, there are only a handful of projects that aim to achieve a similar goal in a decentralized setting. Swarm Parallelism, proposed by [Ryabinin et al.][swarm], is one such method that aims to achieve decentralized training of large models using unreliable heterogeneous devices
with slow interconnect. The results reported in the paper are promising, but their work is still not entirely fit our requirements. The primary concern is the trustless nature of the network. In other words, we cannot blindly trust the participants to perform the computations correctly. Furthermore, the network might come under adversarial attacks that would compromise the integrity of the network which ultimately leads to the failure of the training process.

Ultimately, this approach uses volunteer computing and enables a large number of participants to join forces to train AI models for their desired tasks. Other alternatives include federated learning, a paradigm similar to volunteer computing, but more suitable for large organizations with private data. 

There is a long road ahead, for the first step, we are only focused on ensuring integrity and correctness of the computations for the knowledge distillation task. Once this is achieved, we would be able to utilize the techniques developed to identify and block adversarial participants to train more complex models in a decentralized manner. Ultimately, this subnet will pave the way for the decentralized training of large models and democratize AI research and development.



[swarm]: https://proceedings.mlr.press/v202/ryabinin23a/ryabinin23a.pdf
[zero]: https://arxiv.org/abs/1910.02054
[kd]: https://arxiv.org/abs/1503.02531
[fineweb-blog]: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
