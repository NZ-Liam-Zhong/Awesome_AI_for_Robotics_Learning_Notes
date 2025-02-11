# Learning Notes for Robot Learning (Part 1)
<br>**Skip to** [Part 2](https://github.com/NZ-Liam-Zhong/Awesome_AI_for_Robotics_Learning_Notes_2)<br><br>
As ChatGPT and RT2 appears, many researchers start to do reseaches about AI for robotics. Many scholars have different names for it, including embodied AI, smart robotics, etc. I will update the papers and content that I have read. Unlike the others, I won't just use python scirpt to get the information and abstract, I will share my thoughts after the reading. I promise that very single content listed below I have read it at least once. 
![image](https://github.com/user-attachments/assets/78cd784b-c257-40c0-9296-fda6a717147f)
(Image by Grok)

## 🤔Main Challenges🤔
1. Accuracy (Algorithm) 
2. Generality (Multi-task)
3. Inference speed 
4. Datasets
5. Sim2Real

## Tutorials & Slides
1.[Reinforcement Learning Basics by Fei-Fei Li](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf)
Slides for understanding basic concepts for reinforcement learning. **Institution： Stanford University**

2.[Deep Reinforcement Learning slides by Sergey Levine CS 285 at UC Berkeley](https://rail.eecs.berkeley.edu/deeprlcourse/)
A very good slide to learn the basic concepts of reinforcement learning. Super clear. **Institution： UC Berkeley**
<br> (For the starters)
[Lesson: Deep Reinforcement Learning slides by Sergey Levine CS 285 at UC Berkeley](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps)

3.[CS234: Reinforcement Learning Spring 2024 by Emma Brunskill](https://web.stanford.edu/class/cs234/modules.html)
A very good slide for the learners with a basic RL knowledge. **Institution： Stanford University**

4.[Offline Reinforcement Learning Tutorial by Sergey Levine 2020](https://arxiv.org/pdf/2005.01643)
Introduction, technology and open problem about offline reinforcement learning. **Institution： UC Berkeley**

5.[Impressive Talk by Sergey Levine](https://www.youtube.com/watch?v=mXFH7xs_k_I) Good for grasping the new ideas for future growths in embodied AI. Talk was given in Dec. 2024. <br>1.Generalists are better than specialists. <br>2.Flow matching for robotic control<br> 3.GPT 4o style of reasoning for robotic control<br> 4.Distilling RL knowledge for robotic control <br> **Institution： UC Berkeley**

6.[Learning Tutorials by Sergey Levine](https://drive.google.com/file/d/1_aJxnlwLsJYup-__qKi-ZnujQho6ibDk/view) **Institution： UC Berkeley**

7.[OpenAI tutorials for deep reinforcement learning](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) Tutorials, algorithms and how to learn about reinforcement learning. **Institution： OpenAI**
![图片](https://github.com/user-attachments/assets/bfb1b999-51e4-4087-8995-ca676e4144e0)
<br>1.Vanilla Polivy Gradient (VPG): https://spinningup.openai.com/en/latest/algorithms/vpg.html
<br>2.Trust Region Policy Optimization （TRPO):https: //spinningup.openai.com/en/latest/algorithms/trpo.html (A bit complicated)
<br>3.Proximal Policy Optimization (PPO): https://spinningup.openai.com/en/latest/algorithms/ppo.html<br>
![image](https://github.com/user-attachments/assets/efa07287-c93c-4fc0-8062-7ef327ee3d2b) means we want it to converge more robustly.<br>
![image](https://github.com/user-attachments/assets/7af7bbf4-adb2-41e3-a12a-6aa2408bfcbf)<br>
[Good chinese blog](https://blog.csdn.net/qq_41626059/article/details/114781844?utm_source=chatgpt.com)<br>
<br>4.Deep Deterministic Policy Gradient (DDPG): https://spinningup.openai.com/en/latest/algorithms/ddpg.html (important)
<br>5.Twin Delayed DDPG (TD3):https://spinningup.openai.com/en/latest/algorithms/td3.html (1.2 Q functions choose smaller Q to update 2. Update policy less than Q)
<br>6.Soft Actor Critic (SAC) https://spinningup.openai.com/en/latest/algorithms/sac.html
<br>(1) Unlike in TD3, the target also includes a term that comes from SAC’s use of entropy regularization.
<br>(2) Unlike in TD3, the next-state actions used in the target come from the current policy instead of a target policy.
<br>(3)Unlike in TD3, there is no explicit target policy smoothing. TD3 trains a deterministic policy, and so it accomplishes smoothing by adding random noise to the next-state actions. SAC trains a stochastic policy, and so the noise from that stochasticity is sufficient to get a similar effect.
<br><br>
**What is a "model" based policy?**\
Model is always defined as M where: **Reward, Si+1 = M (Si,a)**


8.[OpenAI's former VP Lilian Weng's technical blog](https://lilianweng.github.io/) Blogs about RL and diffusion models. **Institution： None**

9.Book for Starters \
![图片](https://github.com/user-attachments/assets/b438b3cf-1661-43ba-a944-8f8f6ed9c8c0)\
I have read it. It is good for starters but not as clear as the slides listed above. **Author: Yong Yu,etc**\
Let me remind you that the standard Q learning equation is:\
![图片](https://github.com/user-attachments/assets/6f2c2696-33aa-4bfd-b9c1-028a8b6909fe)\
I will share some algorithms that the other materials havn't mentioned:\
(1)[MPC] It introduces PETS(a kind of MPC) which means we have to have multiple models to sample some and choose the action that maximize the rewards calculated by the models we have chosen. (This book suggests that MPC is a kind of model based method but I think it kind of weired)\
(2)[model based] Dyna-Q is a kind of model based method which uses the trajectory sampled before to learn the model and help to learn the Q function\
(3)[MARL] IPPO is often used when multi-agent share the same policy. Just nothing diferent with PPO and optimize the policy.\
**Offline RL**\
Do you know what's the problems of offline RL?\
Main problem is **Extrapolation Error**.\
**Extrapolation Error** means **model tends to over-estimate the Q values which the model hanvn't seen**. In offline RL the network cannot be fed with new data, so the problem is extremely serious.\
Solutions for that:<br>
1.Batch-constrained Policy **BCQ** (also based on Q learning). For deterministic policy, we only choose the action we have seen. For continous policy, we add noise to the action in policy optimization. \
2.Conservative Q-learning **CQL**: Punish the Q values when over-estimating:\
![图片](https://github.com/user-attachments/assets/adae1dcd-6e95-4f03-be51-36c639b338b0)\
**Goal-oriented RL (GoRL)**\
(1)[HER] When it is hard to have correct trajectory in first few opochs, we can change the goal to fit the wrong trajectory in order to have denser rewards. <br>


10.[Introduction to Model Predictive Control](https://lab.vanderbilt.edu/taha/wp-content/uploads/sites/154/2017/10/EE5243_Module6.pdf)
Given cost and constraint we can solve the best policy for future mutiple steps and apply the first step.**Institution：Caltech** <br>

**Big things in RL/IL Theory in 2024:** <br> 1.Offline RL <br> 2.World Models (Model Based) <br>3.Model Predictive Control <br>4.Multi-agent Reinforcement Learning <br>

11.[Model Predictive Control Lecture](https://web.stanford.edu/class/ee364b/lectures/mpc_slides.pdf) **Institution：Stanford**\
![图片](https://github.com/user-attachments/assets/0891b0da-1a42-46b5-a84e-b7d7147a8a91)

12.[CVPR 2023 diffusion models Tutorials](https://cvpr2023-tutorial-diffusion-models.github.io/) Definitely worth watching, lots of famous reseachers in diffusion model contributed to this repo.

13.[Flow Matching by Meta](https://nips.cc/virtual/2024/tutorial/99531)

14. MARL Notes [learn from this paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf):<br>
(1) Policy gradient (PG) algorithms such as PPO are less sample efficient than off-policy methods. <br>
(2) Value function in PPO is used for variance reduction<br>
(3)MARL algorithms generally fall between two frameworks: centralized and decentralized learning.
Centralized methods directly learn a single policy to produce the joint actions of all agents. In
decentralized learning, each agent optimizes its reward independently; these methods can tackle
general-sum games but may suffer from instability even in simple matrix games. Centralized
training and decentralized execution (CTDE) algorithms fall in between these two frameworks.
Several past CTDE methods adopt actor-critic structures and learn a centralized critic which
takes global information as input. Value-decomposition (VD) methods are another class of CTDE
algorithms which represent the joint Q-function as a function of agents’ local Q-functions
and have established state of the art results in popular MARL benchmarks.<br>
(4)The use of PPO in multi-agent domains is studied by several concurrent works. [7] empirically show
that decentralized, independent PPO (IPPO) can achieve high success rates in several hard SMAC
maps – however, the reported IPPO results remain overall worse than QMix, and the study is limited
to SMAC. [ 25] perform a broad benchmark of various MARL algorithms and note that PPO-based
methods often perform competitively to other methods. Our work, on the other hand, focuses on PPO
and analyzes its performance on a more comprehensive set of cooperative multi-agent benchmarks.
We show PPO achieves strong results in the vast majority of tasks and also identify and analyze
different implementation and hyperparameter factors of PPO which are influential to its performance
multi-agent domains; to the best of our knowledge, these factors have not been studied to this extent
in past work, particularly in multi-agent contexts.<br>
(5)Our empirical analysis of PPO’s implementation and hyperparameter factors in multi-agent settings
is similar to the studies of policy-gradient methods in single-agent RL [34, 17 , 9, 1 ]. We find several
of these suggestions to be useful and include them in our implementation. In our analysis, we focus
on factors that are either largely understudied in the existing literature or are completely unique to the
multi-agent setting.<br>
**This paper tells us that we don't have to improve the algorithm to publish papers, we can just correct a common belief**


15.Master Student from SYSU, **Yixiong Li**, shared nice explanations on diffusion policy. The PPTs listed below are screenshots of his slides. <br>
(1) Discontinuity: <br>
![图片](https://github.com/user-attachments/assets/3b7d01dc-82ba-4311-9900-4c189f73cdea)
(2) Multi-peak distribution，<br>
![图片](https://github.com/user-attachments/assets/be388ec8-de35-4dc8-ba8e-dc7eb9d2dd2a)
(3) Diffusion policy<br>
![图片](https://github.com/user-attachments/assets/2807045d-cde5-4779-ab3b-01f863f4f3ed)
(4)Difference with IBC<br>
![图片](https://github.com/user-attachments/assets/a07305d2-dee2-4d51-b43d-2b0376b6c858)
(5)Final Structure<br>
![图片](https://github.com/user-attachments/assets/b23f162d-8d67-4ffc-94e7-104bfd9aeb90)
![图片](https://github.com/user-attachments/assets/2dd9f952-a661-4e4b-a6a3-28e19305611c)

16. **Big Names on Reinforcement learning.** We can learn from their papers.<br>
David Silver, John Schulman, Sergey Levine, Chelsea Finn, Danijar Hafner, Martin Riedmiller, Marc Bellemare, Rishabh Agarwal, Scott Fujimoto, Ben Van Roy, Stefano Ermon, Jeff Clune, Philip Thomas, Phillip Isola, Tuomas Haarnoja, Deepak Pathak, Abhinav Gupta, Satinder Singh, Doina Precup, Michael Littman, Sutton<br>

17.[Is Value Learning Really the Main Bottleneck in Offline RL?](https://arxiv.org/pdf/2406.09329) Good paper worth reading again and again.<br>
What is the bottleneck?<br> 
(B1) imperfect value function estimation <br>
(B2) imperfect policy extraction guided by the learned value function <br> 
(B3) imperfect policy generalization to states that it will visit during evaluation. <br>
B3 has most severe prblem.<br>
Solution 1: Improve offline data coverage<br>
Solution 2: Test-time policy improvement<br>
(1) On-the-fly policy extraction (OPEX).<br>
(2) Test-time training (TTT).<br>
He has done experiments on <br>
![图片](https://github.com/user-attachments/assets/4d63d661-9b68-47d7-bb52-8bb1693a7f40)
![图片](https://github.com/user-attachments/assets/6261b875-3df8-4e29-8990-ceb936d76fa9)

18.[Diffusion Meets Flow Matching: Two Sides of the Same Coin](https://diffusionflow.github.io/) These slides are easy to comprehend, and it's easy to compare the differences. **Affiliation: Google Deepmind**

19.[Courses on Robot Learning](https://neo-x.github.io/teaching/ift6163) Course about robot learning, mainly devided into RL/IL theory, how to bridge the sim2real gap and the learning the reward function. **Affiliation: Glen Berseth, University De Montreal**

20.Notes on Flow Matching:<br>
Flow Matching is similar to DDIM, in terms of loss, the loss of DDIM is equal to:<br>
![be129779a55d69aaaf2fc4a863405db](https://github.com/user-attachments/assets/585fbaf4-25ad-4cce-929c-dcfba2876bee)<br>
**i) When Noise Intensity is High**<br>

Since the model's estimated original image is defined as:  
![image](https://github.com/user-attachments/assets/048ffa68-7f1d-402c-8dd5-dae1aaaf6d79)<br>

when noise intensity is high (i.e., \(\sigma_t\) is large and \(\alpha_t\) is small), the error between the network's predicted \(\hat{\epsilon}\) and the true noise \(\epsilon\) gets amplified in \(\hat{x}_0\). Common DDIM sampling relies on \(\hat{x}_0\) to calculate the next state at each sampling step, meaning that during the early stages of sampling (when noise intensity is high), the sampling quality is more likely to be affected. 

Moreover, there is an argument that the difficulty for the model to estimate the added noise increases when the noise intensity is high. Essentially, the model is being fed with what is nearly pure noise, and it still has to identify the specific noise components that were added.

In contrast, FM incorporates a weighting term in its loss function,  
![image](https://github.com/user-attachments/assets/f03df80a-9be1-4ed1-b508-5821605417ca)<br>

which increases when noise intensity is high. This effectively raises the penalty during the early sampling stages, encouraging the model to pay greater attention to prediction errors under high noise conditions. As a result, FM may perform better than DM in these stages.

**ii) When Noise Intensity is Low**<br>

When the noise intensity is low, \(\eta\) approaches 0, and the FM loss function becomes nearly identical to that of DM. At first glance, this might not seem advantageous. However, if the prediction target is the original image, FM can still outperform DM by a small margin (even if it's just for show, haha!).

From the earlier derivations, when the prediction target is the original image, the following relationship holds:  
![image](https://github.com/user-attachments/assets/92e5f0f4-eb0b-4a39-96e1-01e6bb946162)<br>

When noise intensity is low (\(\sigma_t\) is small and \(\alpha_t\) is large), the model places less emphasis on noise errors. Especially as \(\sigma_t\) approaches 0, the model tends to ignore prediction errors. Additionally, with  
![image](https://github.com/user-attachments/assets/0862cf04-411c-4b4f-ade4-d16210944bc9)<br>

any prediction errors in \(\hat{x}_0\) are amplified in \(\hat{\epsilon}\), which impacts the sampling quality toward the end of the sampling process (when noise intensity is low). 

In other words, just when the model needs to be more cautious (\(\hat{\epsilon}\) amplifies errors), it pays less attention to errors (as the loss weight decreases). A clear drawback! 

Furthermore, when noise intensity is very low, using the original image as the target may not provide strong guidance to the model. This is because the input to the model is already very similar to the original image, resulting in low information entropy.

21.[DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf) We can learn from the post-training methods from LLM training.
The RL algorithm it used<br>
![image](https://github.com/user-attachments/assets/f53b4afa-d644-4c98-9929-3c72d6aad2a2)
the reward function<br>
![image](https://github.com/user-attachments/assets/23ae04d4-293d-41ff-aaf2-e2bbb8fab824)
the format<br>
![image](https://github.com/user-attachments/assets/2498fb10-246a-48d5-aabd-a803a9708201)

22.[Hunyi Lee's course on machine learning](https://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html) It has a part which focuses on reinforcement learning.
selected slides:<br>
1. Same expetations but not the same variance<br>
![image](https://github.com/user-attachments/assets/f3fa54af-47e0-4fbd-aa9a-29b87efac955)

23.[Slides about Monte Carlo Methods](https://math.arizona.edu/~tgk/mc/)<br>
Among them, [importance sampling chapter](https://math.arizona.edu/~tgk/mc/book_chap6.pdf)<br>
This analysis is important<br>
![image](https://github.com/user-attachments/assets/cca0c7e5-ef3d-4f87-a4da-42b6bca501cf)<br>
theta<br>
![image](https://github.com/user-attachments/assets/25e52355-a3f3-41c8-8916-f1f440850323)

24.[Importance Sampling by CMU](https://www2.stat.duke.edu/~st118/Publication/impsamp.pdf) Importance Sampling: A review<br>

25.[Review on importance Sampling](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7974876) Especially AIS (adaptive Importance Sampling).
Standard PMC [19]: In this algorithm, N proposals are 
adapted via resampling, which is a well-known mechanism 
in MC methodologies that allows us to select the most 
promising samples and to eliminate those with low weights 
to avoid particle degeneracy [29]. At each iteration, exactly 
one sample is drawn from each proposal and weighted 
with the standard IS weights calculated by (15). Then, N 
multinomial resampling steps (with replacement) are per
formed within the population of the N drawn samples (one 
sample is generated per proposal, i.e., K
 1
 = ). The surviv
ing set of particles constitutes the set of location parame
ters for the next population of proposals.<br>
 ■ M-PMC [20]: For this method, the proposal used to gener
ate K samples at each iteration is a mixture of N kernels, 
where the mixture is adapted to decrease the Kullback
Leibler (KL) divergence between the mixture and the 
target. In its simplest version, the algorithm adapts the 
location, scale, and weight of each kernel in the mixture.<br>
 ■ Nonlinear PMC (N-PMC) [32]: In this algorithm, the 
weights are computed in two steps. First, standard impor
tance weights w( )
 j
 k are obtained. Then, a nonlinear func
tion is applied to calculate a set of transformed weights 
{ 
k
 w( )
 j
 .
 The goal of this transformation is to reduce the vari
ance of the weights and avoid, or at least mitigate, the 
IEEE SIgnal ProcESSIng MagazInE   |   July 2017   |
 Authorized licensed use limited to: SUN YAT-SEN UNIVERSITY. Downloaded on February 07,2025 at 06:57:33 UTC from IEEE Xplore.  Restrictions apply. 
67 IEEE SIgnal ProcESSIng MagazInE   |   July 2017   |
 weight degeneracy problem. While the standard weights 
can be used for estimation, the nonlinearly transformed 
weights are crucially used for the adaptation step. The 
 latter can be carried out in different ways, with [32] advo
cating for a simple Gaussian proposal where both the 
mean vector and the covariance matrix are adapted through 
the iterations.<br>
 ■ Layered AIS (LAIS) [23]: The adaptive process of the 
LAIS algorithm is independent of the samples drawn at 
each iteration. In particular, the algorithm can be seen as a 
two-layer procedure in which the location parameters of 
the proposals are adapted through one or several MCMC 
steps with the target as the stationary distribution. In its 
basic version, a single MCMC step is independently per
formed at each location parameter.<br>
 ■ DM-PMC [24]: This algorithm meets the simplicity of 
the standard PMC of [19] with a very high performance. 
DM-PMC calculates the weights using (16) instead of 
(15), which provides two important advantages, specifi
cally, the variance of the estimators is decreased (see 
[25]) and the resampling step with the DM weights pro
motes the replication of proposals in relevant parts of the 
target that are underrepresented by the set of proposals 
(i.e., the exploration is coordinated). DM-PMC generates 
K samples per each of the N proposals (instead of one, 
as in [19]). At each iteration, the  population of KN 
 samples must be reduced to N via either global or local 
resampling (LR).<br>
 ■ AMIS [21]: In this algorithm, just one proposal is used 
and adapted over the iterations. The adaptive procedure 
consists of estimating the moments of the target with the 
available set of K weighted samples and fitting the 
moments of the proposal. Its key feature is the reweight
ing of all of the past samples with a temporal mixture 
weight where the whole sequence of proposals is used in 
the denominator.<br>
 ■ Gradient APIS (GAPIS) [34]: Similar to the LAIS algo
rithm, GAPIS adapts N proposals by a process that is 
independent of the samples. In its basic  version, the loca
tion parameters of the proposals are adapted via a gradient 
ascent of the target and the scale parameter by using the 
Hessian of the target. An advanced implementation is pro
posed that adds a repulsive interaction among proposals to 
promote a cooperative exploration of the target<br>
![image](https://github.com/user-attachments/assets/78a708c3-2712-4255-a52a-f161c79dda16)
![image](https://github.com/user-attachments/assets/99fe6b9b-e292-433f-9f66-6539cb9f89e0)

26.[“A population Monte Carlo scheme with transformed weights and its application to stochastic kinetic models](https://arxiv.org/pdf/1208.5600)

27.[Adaptive Importance Sampling](https://artowen.su.domains/pubtalks/AdaptiveISweb.pdf) Slides by Stanford University
![image](https://github.com/user-attachments/assets/ae938afe-cd25-43d3-a01e-472ab5ebf8e8)<br>
Other Materials:(When the variance will be zero and why)<br>
![image](https://github.com/user-attachments/assets/0f15d2b6-fc9b-4387-8a40-4b90eccc4459)

28.mportance Sampling is a method commonly used to estimate expected values in probability distributions or to perform Monte Carlo simulations, primarily by weighting samples to improve the efficiency of estimation. However, it also has some issues, mainly including the following:<br>

Excessive Weight Variance: The core of importance sampling is to calculate the weighted average of samples, where the weights are typically the ratio of the target distribution to the sampling distribution. If some samples have very large weights while others have very small weights, the overall estimation can be very inaccurate, even leading to a very large variance in the estimation, affecting the stability of the results.<br>

Choosing an Inappropriate Sampling Distribution: If the sampling distribution differs significantly from the target distribution (i.e., the importance sampling ratio varies greatly), the weights of the samples may concentrate on a few points, leading to increased variance. This can make importance sampling inefficient.<br>

Computational Burden: For high-dimensional problems, importance sampling may require a very large number of samples to accurately estimate the expected value. Especially when the target distribution is complex, the calculation of weights can be very time-consuming, further increasing the computational burden.<br>

29.Behaviour Cloning uses loss:<br>
1.MSE (mean value of l2 loss)<br>
2.L1,L2 regularization (L1 more robust, L2 punish outliers harder) <br>
3.cross entropy (classification) <br>

# Paper List
1.Accepted papers in CoRL 2024 are in file “corl2024_paper_list.xlsx"<br>
2.[ICML 2024 Oral paper list](https://icml.cc/virtual/2024/events/oral)
<br> [Reinforcement Learning papers in ICML 2024](https://icml.cc/virtual/2024/papers.html?filter=titles&search=reinforcement+learning)<br>
3.[ICLR 2024 Paper List](https://iclr.cc/virtual/2024/papers.html?filter=titles)<br>



## Papers

### (1) Reinforcement Learning & Imitation Learning
1.[HIQL: Offline Goal-Conditioned RL
with Latent States as Actions](https://proceedings.neurips.cc/paper_files/paper/2023/file/6d7c4a0727e089ed6cdd3151cbe8d8ba-Paper-Conference.pdf)
This paper introduces a new way to improve the performance of action-free offline reinforcement learning. Most of the time, when we want to do training in action-free offline reinforcement learning, we need to train a value network and use it to update a Q function network. However, the authors in this paper have used 2 Q function networks. One of them is used to predict a high level policy, and the other is used to learn a low level policy. **Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine. NeurIPS 2023** 
![image](https://github.com/user-attachments/assets/0f2547b4-4fbb-43a3-9f15-ff863d0e3a05)
![图片](https://github.com/user-attachments/assets/0a8f3d5d-e814-418e-8ca4-1f2509a6f9e2)


2.[Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control](https://arxiv.org/pdf/2401.16889) This paper introduces a multi-stage method for bipedal locomotion control. The network inputs the short term history and the CNN feature of 2 second long history to output the policy. The method has 3 stages. First, it trains a single task by RL. Second, it trains tasks randomly to learn a general policy. Finally, it randomize the dynamics (the parameters of the environments and noise) to bridge the sim2real gap.  **Zhongyu Li, Xue Bin Peng, Pieter Abbeel, Sergey Levine, Glen Berseth, Koushil Sreenath. IJRR 2024** 

3.[Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance](https://openreview.net/pdf?id=6FGlpzC9Po)
This paper learns a value action conditioned on the language guide based on large dataset training. When in reference, we can use the value function the sample the best action in multiple choices. The formula below is how to train the value function Q. **Mitsuhiko Nakamoto, Oier Mees, Aviral Kumar, Sergey Levine. CoRL 2024**
![image](https://github.com/user-attachments/assets/62870f16-21aa-4909-a35e-d07bad3695b7)

4.[Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation](https://arxiv.org/pdf/2403.03949)
A new paradigm for learning robot policy. The authors first use imitation learning to learn the real world policy, reconstructs the environment and uses reinforcement learning to learn the policy in the simulator, and transfer the policy to the real world using teacher-student model (because sim and real have different inputs). I admit that there is performance increase, but I doubt whether it is applicable to the real world particularly in large complicated scenarios.  But we can learn from the paper that, we can add some regularization when learning the strategies so we can adopt the advantages of the previous policy. **Marcel Torne, Anthony Simeonov, Zechu Li, April Chan, Tao Chen, Abhishek Gupta2, Pulkit Agrawal. Arxiv 2024** 

5.[OGBENCH: BENCHMARKING OFFLINE GOAL-CONDITIONED RL](https://arxiv.org/abs/2410.20092)
A benchmark for offline goal-conditioned reinforcement learning to assess the capability of algorithms to deal with challenges in this field. The challenges including (1) Learning from suboptimal, unstructured data (2) Goal stitching (3) Long-horizon reasoning (4) Handling stochasticity, which motivates the design of OGBench. Additionally, OGBench illustrated the open problems for reaearchers to investigate according to the results, which is a foudation for researchers to achieve the goal of offline GCRL:  building foundation models for general-purpose behaviors. **Seohong Park, Kevin Frans, Benjamin Eysenbach, Sergey Levine. Arxiv 2024** 

6.[Avoid Everything: Model-Free Collision Avoidance with Expert-Guided Fine-Tuning](https://openreview.net/forum?id=gqFIybpsLX)The world is full of clutter. In order to operate effectively in uncontrolled, real world spaces, robots must navigate safely by executing tasks around obstacles while in proximity to hazards. Creating safe movement for robotic manipulators remains a long-standing challenge in robotics, particularly in environments with partial observability. In partially observed settings, classical techniques often fail. Learned end-to-end motion policies can infer correct solutions in these settings, but are as-yet unable to produce reliably safe movement when close to obstacles. In this work, we introduce Avoid Everything, a novel end-to-end system for generating collision-free motion toward a target, even targets close to obstacles. Avoid Everything consists of two parts: 1) Motion Policy Transformer (M$\pi$Former), a transformer architecture for end-to-end joint space control from point clouds, trained on over 1,000,000 expert trajectories and 2) a fine-tuning procedure we call Refining on Optimized Policy Experts (ROPE), which uses optimization to provide demonstrations of safe behavior in challenging states. With these techniques, we are able to successfully solve over 63% of reaching problems that caused the previous state of the art method to fail, resulting in an overall success rate of over 91% in challenging manipulation settings. **Adam Fishman,Aaron Walsman,Mohak Bhardwaj,Wentao Yuan,Balakumar Sundaralingam,Byron Boots,Dieter Fox CoRL2024**

7.[Hard Tasks First: Multi-Task Reinforcement Learning Through Task Scheduling](https://raw.githubusercontent.com/mlresearch/v235/main/assets/cho24d/cho24d.pdf)Multi-task reinforcement learning (RL) faces the
significant challenge of varying task difficulties,
often leading to negative transfer when simpler
tasks overshadow the learning of more complex
ones. To overcome this challenge, we propose a
novel algorithm, Scheduled Multi-Task Training
(SMT), that strategically prioritizes more chal-
lenging tasks, thereby enhancing overall learning
efficiency. SMT introduces a dynamic task pri-
oritization strategy, underpinned by an effective
metric for assessing task difficulty. This metric en-
sures an efficient and targeted allocation of train-
ing resources, significantly improving learning
outcomes. Additionally, SMT incorporates a reset
mechanism that periodically reinitializes key net-
work parameters to mitigate the simplicity bias,
further enhancing the adaptability and robustness
of the learning process across diverse tasks. The
efficacy of SMT’s scheduling method is validated
by significantly improving performance on chal-
lenging Meta-World benchmarks. **Myungsik Cho Jongeui Park Suyoung Lee Youngchul Sung ICML 2024**
![图片](https://github.com/user-attachments/assets/0a605d5a-d1b6-4316-b98c-8d76e6782df4)



### (2) Vision-Language Action Model 
1.[MiniVLA: A Better VLA with a Smaller Footprint](https://ai.stanford.edu/blog/minivla/)
The OpenVLA by far the sota method in open-source vision-language action models. However, the model has a slow inference and training speed and only supprt per-image input. In this work, the authors use action chunking (multi-action output) and multi-image input to improve the performance. What's more, the authors also use Qwen 0.5B model as the backbone to reduce the size of the backbone mode. **Suneel Belkhale and Dorsa Sadigh. The Stanford AI Lab Blog 2024** 

2.[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf) Typically, we need RLHF to finetune any LLM to human preference datasets. However, it needs fitting an Bradley-Terry Model based reward function. This paper uses simple algebra to put foward a loss function that can directly optimize the LLM without learning any reward model (The model itself can be used as reward model). What we can learn from this paper is that some learning frameworks can be optimized through simple algebra.**Rafael Rafailov, Archit Sharma, Eric Mitchell,Stefano Ermon, Christopher D. Manning, Chelsea Finn. NeurIPS 2023** 
![图片](https://github.com/user-attachments/assets/1ec6ead7-1fee-4112-9df8-e08517de452c)
![图片](https://github.com/user-attachments/assets/cee92256-2cfe-46c3-aaa4-5d6466062820)

3.[ReST-MCTS∗: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/pdf/2406.03816) A technical method hopes to realize the slow-thinking mode of GPT o1. It uses tree searching method to train both the value network and policy network.**Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, Jie Tang. Arxiv 2024**
![图片](https://github.com/user-attachments/assets/30662c8d-379e-4a6b-a982-212b92818ce2)
![图片](https://github.com/user-attachments/assets/7fa751bd-99c8-4c0d-9951-ac36907c9fca)

4.[Robotic Control via Embodied Chain-of-Thought Reasoning](https://openreview.net/forum?id=S70MgnIA0v) In the field of nature language processing, chain-of-thought (CoT) improves the performance of large language model quite a lot. This paper investigates whether this method can improve the performance of Embodied AI. However, embodied environments and tasks are more complex than that in large language model. So, "embodied" attribute should be added to CoT (ECoT), which aims to answer the following questions: Q1: Which reasoning steps are suitable for guiding policies in solving embodied robot manipulation tasks? A1: The ECoT contains TASK, PLAN, SUBTASK, MOVE, GRIPPER POSITION, VISIBLE OGJECTS, which not only focuses on "how to think", but also focuses on "how to look". Q2: How to generate datasets for reasoning? A2: generate the data by a pipeline using Prismatic VLM, Grounding DINO, OWL and SAM. Q3: How to increase the speed while reasoning? This question is for future work. The experiments show that ECoT improves the generation performance compared with OpenVLA and RT2X.**Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, Sergey Levine. CoRL 2024**

5.[Autonomous Improvement of Instruction Following Skills via Foundation Models](https://arxiv.org/pdf/2407.20635)Intelligent instruction-following robots capable of improving from au-
tonomously collected experience have the potential to transform robot learning:
instead of collecting costly teleoperated demonstration data, large-scale deploy-
ment of fleets of robots can quickly collect larger quantities of autonomous data
that can collectively improve their performance. However, autonomous improve-
ment requires solving two key problems: (i) fully automating a scalable data col-
lection procedure that can collect diverse and semantically meaningful robot data
and (ii) learning from non-optimal, autonomous data with no human annotations.
To this end, we propose a novel approach that addresses these challenges, allow-
ing instruction-following policies to improve from autonomously collected data
without human supervision. Our framework leverages vision-language models to
collect and evaluate semantically meaningful experiences in new environments,
and then utilizes a decomposition of instruction following tasks into (semantic)
language-conditioned image generation and (non-semantic) goal reaching, which
makes it significantly more practical to improve from this autonomously collected
data without any human annotations. We carry out extensive experiments in the
real world to demonstrate the effectiveness of our approach, and find that in a suite
of unseen environments, the robot policy can be improved 2x with autonomously
collected data. We open-source the code for our semantic autonomous improve-
ment pipeline, as well as our autonomous dataset of 30.5K trajectories collected
across five tabletop environments. **Zhiyuan Zhou, Pranav Atreya, Abraham Lee, Homer Walke, Oier Mees, Sergey Levine CoRL2024**

6.[LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning](https://openreview.net/forum?id=Q2lGXMZCv8)In recent years, instruction-tuned Large Multimodal Models (LMMs) have been successful at several tasks, including image captioning and visual question answering; yet leveraging these models remains an open question for robotics. Prior LMMs for robotics applications have been extensively trained on language and action data, but their ability to generalize in different settings has often been less than desired. To address this, we introduce LLARVA, a model trained with a novel instruction tuning method that leverages structured prompts to unify a range of robotic learning tasks, scenarios, and environments. Additionally, we show that predicting intermediate 2-D representations, which we refer to as visual traces, can help further align vision and action spaces for robot learning. We generate 8.5M image-visual trace pairs from the Open X-Embodiment dataset in order to pre-train our model, and we evaluate on 12 different tasks in the RLBench simulator as well as a physical Franka Emika Panda 7-DoF robot. Our experiments yield strong performance, demonstrating that LLARVA — using 2-D and language representations — performs well compared to several contemporary baselines, and can generalize across various robot environments and configurations. **Dantong Niu,Yuvan Sharma,Giscard Biamby,Jerome Quenum,Yutong Bai,Baifeng Shi,Trevor Darrell,Roei Herzig CoRL2024**

7.[GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://proceedings.mlr.press/v162/du22c/du22c.pdf)GLaM (Generalist Language Model) is a sparse Mixture-of-Experts (MoE) language model that significantly improves the efficiency and scalability of large-scale language models. By activating only a small subset of parameters (e.g., 8% in the largest 1.2 trillion-parameter version) during inference, GLaM achieves superior performance in zero-shot, one-shot, and few-shot tasks compared to GPT-3, while reducing training energy consumption by two-thirds and halving inference costs. Its dynamic gating mechanism efficiently selects specialized experts for each input, making GLaM a more sustainable and versatile solution for advancing large-scale AI systems. (I haven't read the code) **Nan Du, etc ICML 2022**
![图片](https://github.com/user-attachments/assets/c1ad61f8-17f8-4d90-aaa9-ded1f0339ac2)
<br>**Expert: MoE based models are harder to converge**

8.[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/)
under reading

9.[Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/pdf/2310.08864)
under reading

10.[π0: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/download/pi0.pdf)
under reading

11.[OpenVLA: An Open-Source Vision-Language-Action Model](https://openreview.net/forum?id=ZMnD6QZAE6)
under reading

12.[RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning](https://generalist-distillation.github.io/static/high_performance_generalist.pdf)
under reading

13.[FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747)
Even though the traditional tokenizer of action is efficient on low-frequency tasks, it cannot handle highly dexterous and high-frequency tasks. This paper introduced a new tokenizer for continuous action based on compression method. It use DCT to transfer the action from time domain to frequency domain, and then use scale-and-round method to quantize the frequency compoents, flatten the frequency matrix to action sequence (low-frequency components first), and train a BPE network to losslessly compress the sequence into action token. The new tokenizer dramatically decrease the training time compared with pi0 and openvla and remarkably handle the dexterous tasks. **Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, Sergey Levine. arxiv** 
### (3) Dataset & Dataset Generation
1.[Automated Creation of Digital Cousins for Robust Policy Learning](https://arxiv.org/pdf/2410.07408)
This is a kind of domain randomization method. When training a robot policy based on imgaes input, the authors leverage the depth information and the basic layouts to match different kinds of similar objects to replace the original objects in the scene (The authors defind it as the "digital cousin") and render the new objects to form a new image. When training on different kinds of objects in the same layout, we can bridge the sim2real domain gap. **Tianyuan Dai, Josiah Wong, Yunfan Jiang, Chen Wang, Cem Gokmen, Ruohan Zhang, Jiajun Wu, Li Fei-Fei CoRL 2024** 
![图片](https://github.com/user-attachments/assets/f85388b0-4bf7-4ddf-9874-74e83dc095bc)


2.[RP1M: A Large-Scale Motion Dataset for Piano Playing with Bi-Manual Dexterous Robot Hands](https://openreview.net/forum?id=4Of4UWyBXE)Endowing robot hands with human-level dexterity is a long-lasting research objective. Bi-manual robot piano playing constitutes a task that combines challenges from dynamic tasks, such as generating fast while precise motions, with slower but contact-rich manipulation problems. Although reinforcement learning based approaches have shown promising results in single-task performance, these methods struggle in a multi-song setting. Our work aims to close this gap and, thereby, enable imitation learning approaches for robot piano playing at scale. To this end, we introduce the Robot Piano 1 Million (RP1M) dataset, containing bi-manual robot piano playing motion data of more than one million trajectories. We formulate finger placements as an optimal transport problem, thus, enabling automatic annotation of vast amounts of unlabeled songs. Benchmarking existing imitation learning approaches shows that such approaches reach state-of-the-art robot piano playing performance by leveraging RP1M. **Yi Zhao,Le Chen,Jan Schneider,Quankai Gao,Juho Kannala,Bernhard Schölkopf,Joni Pajarinen,Dieter Büchler CoRL 2024**

### (4) Inference Speed 
1. [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
Speculative Execution: an optimization technique, where a task is performed in parrallel, can accelerate inference without changing the model architecture, withing re-training, without changing the training procedures and without changing the model output distributions. **Yaniv Leviathan, Matan Kalman, Yossi Matias ICML 2023**
![image](https://github.com/user-attachments/assets/41a6eb6b-4e0d-4c24-90c9-eb64f644dfb2)

### (5) Scene Representation
1.[GraspSplats: Efficient Manipulation with 3D Feature Splatting](https://openreview.net/forum?id=pPhTsonbXq) The ability for robots to perform efficient and zero-shot grasping of object parts is crucial for practical applications and is becoming prevalent with recent advances in Vision-Language Models (VLMs). To bridge the 2D-to-3D gap for representations to support such a capability, existing methods rely on neural fields (NeRFs) via differentiable rendering or point-based projection methods. However, we demonstrate that NeRFs are inappropriate for scene changes due to its implicitness and point-based methods are inaccurate for part localization without rendering-based optimization. To amend these issues, we propose GraspSplats. Using depth supervision and a novel reference feature computation method, GraspSplats can generate high-quality scene representations under 60 seconds. We further validate the advantages of Gaussian-based representation by showing that the explicit and optimized geometry in GraspSplats is sufficient to natively support (1) real-time grasp sampling and (2) dynamic and articulated object manipulation with point trackers. With extensive experiments on a Franka robot, we demonstrate that GraspSplats significantly outperforms existing methods under diverse task settings. In particular, GraspSplats outperforms NeRF-based methods like F3RM and LERF-TOGO, and 2D detection methods. The code will be released. **Mazeyu Ji,Ri-Zhao Qiu,Xueyan Zou,Xiaolong Wang CoRL 2024**

2.[Robot See Robot Do: Imitating Articulated Object Manipulation with Monocular 4D Reconstruction](https://openreview.net/forum?id=2LLu3gavF1)Humans can learn to manipulate new objects by simply watching others; providing robots with the ability to learn from such demonstrations would enable a natural interface specifying new behaviors. This work develops Robot See Robot Do (RSRD), a method for imitating articulated object manipulation from a single monocular RGB human demonstration given a single static multi- view object scan. We first propose 4D Differentiable Part Models (4D-DPM), a method for recovering 3D part motion from a monocular video with differentiable rendering. This analysis-by-synthesis approach uses part-centric feature fields in an iterative optimization which enables the use of geometric regularizers to re- cover 3D motions from only a single video. Given this 4D reconstruction, the robot replicates object trajectories by planning bimanual arm motions that induce the demonstrated object part motion. By representing demonstrations as part- centric trajectories, RSRD focuses on replicating the demonstration’s intended behavior while considering the robot’s own morphological limits, rather than at- tempting to reproduce the hand’s motion. We evaluate 4D-DPM’s 3D tracking accuracy on ground truth annotated 3D part trajectories and RSRD’s physical ex- ecution performance on 9 objects across 10 trials each on a bimanual YuMi robot. Each phase of RSRD achieves an average of 87% success rate, for a total end- to-end success rate of 60% across 90 trials. Notably, this is accomplished using only feature fields distilled from large pretrained vision models — without any task-specific training, fine-tuning, dataset collection, or annotation. Project page: https://robot-see-robot-do.github.io **Justin Kerr,Chung Min Kim,Mingxuan Wu,Brent Yi,Qianqian Wang,Ken Goldberg,Angjoo Kanazawa CoRL2024**

3.[Gaussian Splatting to Real World Flight Navigation Transfer with Liquid Networks](https://openreview.net/forum?id=ubq7Co6Cbv)Simulators are powerful tools for autonomous robot learning as they offer scalable data generation, flexible design, and optimization of trajectories. However, transferring behavior learned from simulation data into the real world proves to be difficult, usually mitigated with compute-heavy domain randomization methods or further model fine-tuning. We present a method to improve generalization and robustness to distribution shifts in sim-to-real visual quadrotor navigation tasks. To this end, we first build a simulator by integrating Gaussian Splatting with quadrotor flight dynamics, and then, train robust navigation policies using Liquid neural networks. In this way, we obtain a full-stack imitation learning protocol that combines advances in 3D Gaussian splatting radiance field rendering, crafty programming of expert demonstration training data, and the task understanding capabilities of Liquid networks. Through a series of quantitative flight tests, we demonstrate the robust transfer of navigation skills learned in a single simulation scene directly to the real world. We further show the ability to maintain performance beyond the training environment under drastic distribution and physical environment changes. Our learned Liquid policies, trained on single target maneuvers curated from a photorealistic simulated indoor flight only, generalize to multi-step hikes onboard a real hardware platform outdoors. **Alex Quach,Makram Chahine,Alexander Amini,Ramin Hasani,Daniela Rus CoRL2024**

4.[Physically Embodied Gaussian Splatting: A Visually Learnt and Physically Grounded 3D Representation for Robotics](https://openreview.net/forum?id=AEq0onGrN2)For robots to robustly understand and interact with the physical world, it is highly beneficial to have a comprehensive representation -- modelling geometry, physics, and visual observations -- that informs perception, planning, and control algorithms. We propose a novel dual "Gaussian-Particle" representation that models the physical world while (i) enabling predictive simulation of future states and (ii) allowing online correction from visual observations in a dynamic world. Our representation comprises particles that capture the geometrical aspect of objects in the world and can be used alongside a particle-based physics system to anticipate physically plausible future states. Attached to these particles are 3D Gaussians that render images from any viewpoint through a splatting process thus capturing the visual state. By comparing the predicted and observed images, our approach generates "visual forces" that correct the particle positions while respecting known physical constraints. By integrating predictive physical modeling with continuous visually-derived corrections, our unified representation reasons about the present and future while synchronizing with reality. We validate our approach on 2D and 3D tracking tasks as well as photometric reconstruction quality. Videos are found at https://embodied-gaussians.github.io/ **Jad Abou-Chakra,Krishan Rana,Feras Dayoub,Niko Suenderhauf CoRL2024**

5.[Event3DGS: Event-Based 3D Gaussian Splatting for High-Speed Robot Egomotion](https://openreview.net/forum?id=EyEE7547vy)By combining differentiable rendering with explicit point-based scene representations, 3D Gaussian Splatting (3DGS) has demonstrated breakthrough 3D reconstruction capabilities. However, to date 3DGS has had limited impact on robotics, where high-speed egomotion is pervasive: Egomotion introduces motion blur and leads to artifacts in existing frame-based 3DGS reconstruction methods. To address this challenge, we introduce Event3DGS, an event-based 3DGS framework. By exploiting the exceptional temporal resolution of event cameras, Event3GDS can reconstruct high-fidelity 3D structure and appearance under high-speed egomotion. Extensive experiments on multiple synthetic and real-world datasets demonstrate the superiority of Event3DGS compared with existing event-based dense 3D scene reconstruction frameworks; Event3DGS substantially improves reconstruction quality (+3dB) while reducing computational costs by 95%. Our framework also allows one to incorporate a few motion-blurred frame-based measurements into the reconstruction process to further improve appearance fidelity without loss of structural accuracy. **Tianyi Xiong,Jiayi Wu,Botao He,Cornelia Fermuller,Yiannis Aloimonos,Heng Huang,Christopher Metzler CoRL2024**

### (6) Generative Models
**Generative Models are important for robot policy**, we can either use them as tools or understand them deeply<br>
1.[Dreamitate: Real-World Visuomotor Policy Learning via Video Generation](https://openreview.net/forum?id=InT87E5sr4)A key challenge in manipulation is learning a policy that can robustly generalize to diverse visual environments. A promising mechanism for learning robust policies is to leverage video generative models, which are pretrained on large-scale datasets of internet videos. In this paper, we propose a visuomotor policy learning framework that fine-tunes a video diffusion model on human demonstrations of a given task. At test time, we generate an example of an execution of the task conditioned on images of a novel scene, and use this synthesized execution directly to control the robot. Our key insight is that using common tools allows us to effortlessly bridge the embodiment gap between the human hand and the robot manipulator. We evaluate our approach on 4 tasks of increasing complexity and demonstrate that capitalizing on internet-scale generative models allows the learned policy to achieve a significantly higher degree of generalization than existing behavior cloning approaches. **Junbang Liang,Ruoshi Liu,Ege Ozguroglu,Sruthi Sudhakar,Achal Dave,Pavel Tokmakov,Shuran Song,Carl Vondrick CoRL 2024**

2.[InstructPix2Pix: Learning to Follow Image Editing Instructions](https://www.doubao.com/chat/794004774763266)We propose a method for editing images from human in-
structions: given an input image and a written instruction
that tells the model what to do, our model follows these in-
structions to edit the image. To obtain training data for
this problem, we combine the knowledge of two large pre-
trained models—a language model (GPT-3) and a text-to-
image model (Stable Diffusion)—to generate a large dataset
of image editing examples. Our conditional diffusion model,
InstructPix2Pix, is trained on our generated data, and gen-
eralizes to real images and user-written instructions at in-
ference time. Since it performs edits in the forward pass and
does not require per-example fine-tuning or inversion, our
model edits images quickly, in a matter of seconds. We show
compelling editing results for a diverse collection of input
images and written instructions.<br>
![图片](https://github.com/user-attachments/assets/bd3ef3b7-2b3e-4908-9425-edfceac0a73a)



### (7) Sim2Real
1.[TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction](https://openreview.net/forum?id=lpjPft4RQT)Learning in simulation and transferring the learned policy to the real world has the potential to enable generalist robots. The key challenge of this approach is to address simulation-to-reality (sim-to-real) gaps. Previous methods often require domain-specific knowledge a priori. We argue that a straightforward way to obtain such knowledge is by asking humans to observe and assist robot policy execution in the real world. The robots can then learn from humans to close various sim-to-real gaps. We propose TRANSIC, a data-driven approach to enable successful sim-to-real transfer based on a human-in-the-loop framework. TRANSIC allows humans to augment simulation policies to overcome various unmodeled sim-to-real gaps holistically through intervention and online correction. Residual policies can be learned from human corrections and integrated with simulation policies for autonomous execution. We show that our approach can achieve successful sim-to-real transfer in complex and contact-rich manipulation tasks such as furniture assembly. Through synergistic integration of policies learned in simulation and from humans, TRANSIC is effective as a holistic approach to addressing various, often coexisting sim-to-real gaps. It displays attractive properties such as scaling with human effort. Videos and code are available at https://transic-robot.github.io/. **Yunfan Jiang,Chen Wang,Ruohan Zhang,Jiajun Wu,Li Fei-Fei CoRL2024**

### (8)Visual Representation

1.[What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?](https://openreview.net/forum?id=A1hpY5RNiH)Inspired by the success of transfer learning in computer vision, roboticists have investigated visual pre-training as a means to improve the learning efficiency and generalization ability of policies learned from pixels. To that end, past work has favored large object interaction datasets, such as first-person videos of humans completing diverse tasks, in pursuit of manipulation-relevant features. Although this approach improves the efficiency of policy learning, it remains unclear how reliable these representations are in the presence of distribution shifts that arise commonly in robotic applications. Surprisingly, we find that visual representations designed for control tasks do not necessarily generalize under subtle changes in lighting and scene texture or the introduction of distractor objects. To understand what properties do lead to robust representations, we compare the performance of 15 pre-trained vision models under different visual appearances. We find that emergent segmentation ability is a strong predictor of out-of-distribution generalization among ViT models. The rank order induced by this metric is more predictive than metrics that have previously guided generalization research within computer vision and machine learning, such as downstream ImageNet accuracy, in-domain accuracy, or shape-bias as evaluated by cue-conflict performance. We test this finding extensively on a suite of distribution shifts in ten tasks across two simulated manipulation environments. On the ALOHA setup, segmentation score predicts real-world performance after offline training with 50 demonstrations.**Kaylee Burns,Zach Witzel,Jubayer Ibn Hamid,Tianhe Yu,Chelsea Finn,Karol Hausman CoRL 2024**

### (9) Multi-Robot & Multi-agent Reinforcement Learning (MARL)
1.[CoViS-Net: A Cooperative Visual Spatial Foundation Model for Multi-Robot Applications](https://openreview.net/forum?id=KULBk5q24a)Autonomous robot operation in unstructured environments is often underpinned by spatial understanding through vision. Systems composed of multiple concurrently operating robots additionally require access to frequent, accurate and reliable pose estimates. Classical vision-based methods to regress relative pose are commonly computationally expensive (precluding real-time applications), and often lack data-derived priors for resolving ambiguities. In this work, we propose CoViS-Net, a cooperative, multi-robot visual spatial foundation model that learns spatial priors from data, enabling pose estimation as well as general spatial comprehension. Our model is fully decentralized, platform-agnostic, executable in real-time using onboard compute, and does not require existing networking infrastructure. CoViS-Net provides relative pose estimates and a local bird's-eye-view (BEV) representation, even without camera overlap between robots, and can predict BEV representations of unseen regions. We demonstrate its use in a multi-robot formation control task across various real-world settings. We provide supplementary material online and will open source our trained model in due course. https://sites.google.com/view/covis-net **Jan Blumenkamp,Steven Morad,Jennifer Gielis,Amanda Prorok CoRL2024**<br>
MARL has two types, Centralized Learning and Decentralized Learning.
<br>
**(1) Centralized Learning**
<br>

**(2) Decentralized Learning** <br>

## Industry
Here we update some tech advance in the industry🔥🔥🔥
There are too many start-ups doing the same thing, boring and useless. So, we will only update the AI for robotic start-ups who make **innovative products**

1.[Clone AI](https://www.clonerobotics.com/)
Cloning human beings' body structure to make a humanoid robot.
![image](https://github.com/user-attachments/assets/3a750590-e473-4ce3-a60e-d55d30996af6)

2.[Tesla Optimus](https://x.com/Tesla_Optimus)
The renown Tesla robotics. AGI for robotic control. 
![image](https://github.com/user-attachments/assets/82d682bc-9229-4b94-affd-6a028fbcd162)

3.[Physical Intelligence](https://www.physicalintelligence.company/)
An algorithm company. Designing AI algorithms for robotics. Its founders include lots of renown scholars in robotic area.
![image](https://github.com/user-attachments/assets/99f16bfa-4382-4609-a835-1406d5b7f565)


## 🖊Notes🖊

### Jan.2025
1.Using H20 for OpenVLA LIBERO simulation will have floating point problems. I have sent the solving methods to the "pull requests" in OpenVLA repo.

### Nov.2024
1.Bridge Sim2Real Gap: domain randomization, system identification, or improved simulator visuals <br>
2.When connecting to huggingface is not convenient, we can set: **os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'**<br>
3.When "labels" in "CausalLMOutputWithPast" is -100, it means we can neglect it.


