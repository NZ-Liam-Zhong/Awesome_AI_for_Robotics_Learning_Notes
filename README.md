# Awesome_AI_for_Robotics
As ChatGPT and RT2 appears, many reseachers start to do reseaches about AI for robotics. Many scholars have different names for it, including embodied AI, smart robotics, etc. I don't care what its name really is, I just want to share what I have learnt. Hope it helps!


##Tutorials & Slides
1. [slides by Sergey Levine (UC Berkeley)](https://rail.eecs.berkeley.edu/deeprlcourse/)



##Papers and Abstract
1.[HIQL: Offline Goal-Conditioned RL
with Latent States as Actions UC Berkeley&Princeton](https://proceedings.neurips.cc/paper_files/paper/2023/file/6d7c4a0727e089ed6cdd3151cbe8d8ba-Paper-Conference.pdf)
This paper introduces a new way to improve the performance of action-free offline reinforcement learning. Most of the time, when we want to do training in action-free offline reinforcement learning, we need to train a value network and use it to update a Q function network. However, the authors in this paper have used 2 Q function networks. One of them is used to predict a high level policy, and the other is used to learn a low level policy.

