# Awesome_AI_for_Robotics
As ChatGPT and RT2 appears, many researchers start to do reseaches about AI for robotics. Many scholars have different names for it, including embodied AI, smart robotics, etc. I will update the papers and content that I have read. Unlike the others, I will not just use python scirpt to get the information and abstract, I will share my thoughts after the reading. I promise that very single content listed below I have read it at least once. 
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
<br>3.Proximal Policy Optimization (PPO): https://spinningup.openai.com/en/latest/algorithms/ppo.html
<br>4.Deep Deterministic Policy Gradient (DDPG): https://spinningup.openai.com/en/latest/algorithms/ddpg.html (important)
<br>5.Twin Delayed DDPG (TD3):https://spinningup.openai.com/en/latest/algorithms/td3.html (1.2 Q functions choose smaller Q to update 2. Update policy less than Q)
<br>6.Soft Actor Critic (SAC) https://spinningup.openai.com/en/latest/algorithms/sac.html
<br>(1) Unlike in TD3, the target also includes a term that comes from SAC’s use of entropy regularization.
<br>(2) Unlike in TD3, the next-state actions used in the target come from the current policy instead of a target policy.
<br>(3)Unlike in TD3, there is no explicit target policy smoothing. TD3 trains a deterministic policy, and so it accomplishes smoothing by adding random noise to the next-state actions. SAC trains a stochastic policy, and so the noise from that stochasticity is sufficient to get a similar effect.


8.[OpenAI's former VP Lilian Weng's technical blog](https://lilianweng.github.io/) Blogs about RL and diffusion models. **Institution： None**

## Paper List
1.Accepted papers in CoRL 2024 are in file “corl2024_paper_list.xlsx"<br>
2.[ICML 2024 Oral paper list](https://icml.cc/virtual/2024/events/oral)
<br> [Reinforcement Learning papers in ICML 2024](https://icml.cc/virtual/2024/papers.html?filter=titles&search=reinforcement+learning)



## Papers

### (1) Reinforcement Learning & Imitation Learning
1.[HIQL: Offline Goal-Conditioned RL
with Latent States as Actions](https://proceedings.neurips.cc/paper_files/paper/2023/file/6d7c4a0727e089ed6cdd3151cbe8d8ba-Paper-Conference.pdf)
This paper introduces a new way to improve the performance of action-free offline reinforcement learning. Most of the time, when we want to do training in action-free offline reinforcement learning, we need to train a value network and use it to update a Q function network. However, the authors in this paper have used 2 Q function networks. One of them is used to predict a high level policy, and the other is used to learn a low level policy. **Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine. NeurIPS 2023** 
![image](https://github.com/user-attachments/assets/0f2547b4-4fbb-43a3-9f15-ff863d0e3a05)


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

5.[Autonomous Improvement of Instruction Following Skills via Foundation Models](https://arxiv.org/pdf/2407.20635)
UNDER READING

6.[LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning](https://openreview.net/forum?id=Q2lGXMZCv8)In recent years, instruction-tuned Large Multimodal Models (LMMs) have been successful at several tasks, including image captioning and visual question answering; yet leveraging these models remains an open question for robotics. Prior LMMs for robotics applications have been extensively trained on language and action data, but their ability to generalize in different settings has often been less than desired. To address this, we introduce LLARVA, a model trained with a novel instruction tuning method that leverages structured prompts to unify a range of robotic learning tasks, scenarios, and environments. Additionally, we show that predicting intermediate 2-D representations, which we refer to as visual traces, can help further align vision and action spaces for robot learning. We generate 8.5M image-visual trace pairs from the Open X-Embodiment dataset in order to pre-train our model, and we evaluate on 12 different tasks in the RLBench simulator as well as a physical Franka Emika Panda 7-DoF robot. Our experiments yield strong performance, demonstrating that LLARVA — using 2-D and language representations — performs well compared to several contemporary baselines, and can generalize across various robot environments and configurations. **Dantong Niu,Yuvan Sharma,Giscard Biamby,Jerome Quenum,Yutong Bai,Baifeng Shi,Trevor Darrell,Roei Herzig CoRL2024**

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
1.[Dreamitate: Real-World Visuomotor Policy Learning via Video Generation](https://openreview.net/forum?id=InT87E5sr4)A key challenge in manipulation is learning a policy that can robustly generalize to diverse visual environments. A promising mechanism for learning robust policies is to leverage video generative models, which are pretrained on large-scale datasets of internet videos. In this paper, we propose a visuomotor policy learning framework that fine-tunes a video diffusion model on human demonstrations of a given task. At test time, we generate an example of an execution of the task conditioned on images of a novel scene, and use this synthesized execution directly to control the robot. Our key insight is that using common tools allows us to effortlessly bridge the embodiment gap between the human hand and the robot manipulator. We evaluate our approach on 4 tasks of increasing complexity and demonstrate that capitalizing on internet-scale generative models allows the learned policy to achieve a significantly higher degree of generalization than existing behavior cloning approaches. **Junbang Liang,Ruoshi Liu,Ege Ozguroglu,Sruthi Sudhakar,Achal Dave,Pavel Tokmakov,Shuran Song,Carl Vondrick CoRL 2024**

### (7) Sim2Real
1.[TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction](https://openreview.net/forum?id=lpjPft4RQT)Learning in simulation and transferring the learned policy to the real world has the potential to enable generalist robots. The key challenge of this approach is to address simulation-to-reality (sim-to-real) gaps. Previous methods often require domain-specific knowledge a priori. We argue that a straightforward way to obtain such knowledge is by asking humans to observe and assist robot policy execution in the real world. The robots can then learn from humans to close various sim-to-real gaps. We propose TRANSIC, a data-driven approach to enable successful sim-to-real transfer based on a human-in-the-loop framework. TRANSIC allows humans to augment simulation policies to overcome various unmodeled sim-to-real gaps holistically through intervention and online correction. Residual policies can be learned from human corrections and integrated with simulation policies for autonomous execution. We show that our approach can achieve successful sim-to-real transfer in complex and contact-rich manipulation tasks such as furniture assembly. Through synergistic integration of policies learned in simulation and from humans, TRANSIC is effective as a holistic approach to addressing various, often coexisting sim-to-real gaps. It displays attractive properties such as scaling with human effort. Videos and code are available at https://transic-robot.github.io/. **Yunfan Jiang,Chen Wang,Ruohan Zhang,Jiajun Wu,Li Fei-Fei CoRL2024**

### (8)Visual Representation

1.[What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?](https://openreview.net/forum?id=A1hpY5RNiH)Inspired by the success of transfer learning in computer vision, roboticists have investigated visual pre-training as a means to improve the learning efficiency and generalization ability of policies learned from pixels. To that end, past work has favored large object interaction datasets, such as first-person videos of humans completing diverse tasks, in pursuit of manipulation-relevant features. Although this approach improves the efficiency of policy learning, it remains unclear how reliable these representations are in the presence of distribution shifts that arise commonly in robotic applications. Surprisingly, we find that visual representations designed for control tasks do not necessarily generalize under subtle changes in lighting and scene texture or the introduction of distractor objects. To understand what properties do lead to robust representations, we compare the performance of 15 pre-trained vision models under different visual appearances. We find that emergent segmentation ability is a strong predictor of out-of-distribution generalization among ViT models. The rank order induced by this metric is more predictive than metrics that have previously guided generalization research within computer vision and machine learning, such as downstream ImageNet accuracy, in-domain accuracy, or shape-bias as evaluated by cue-conflict performance. We test this finding extensively on a suite of distribution shifts in ten tasks across two simulated manipulation environments. On the ALOHA setup, segmentation score predicts real-world performance after offline training with 50 demonstrations.**Kaylee Burns,Zach Witzel,Jubayer Ibn Hamid,Tianhe Yu,Chelsea Finn,Karol Hausman CoRL 2024**

### (9) Multi-Robot & Multi-agent Reinforcement Learning (MARL)
1.[CoViS-Net: A Cooperative Visual Spatial Foundation Model for Multi-Robot Applications](https://openreview.net/forum?id=KULBk5q24a)Autonomous robot operation in unstructured environments is often underpinned by spatial understanding through vision. Systems composed of multiple concurrently operating robots additionally require access to frequent, accurate and reliable pose estimates. Classical vision-based methods to regress relative pose are commonly computationally expensive (precluding real-time applications), and often lack data-derived priors for resolving ambiguities. In this work, we propose CoViS-Net, a cooperative, multi-robot visual spatial foundation model that learns spatial priors from data, enabling pose estimation as well as general spatial comprehension. Our model is fully decentralized, platform-agnostic, executable in real-time using onboard compute, and does not require existing networking infrastructure. CoViS-Net provides relative pose estimates and a local bird's-eye-view (BEV) representation, even without camera overlap between robots, and can predict BEV representations of unseen regions. We demonstrate its use in a multi-robot formation control task across various real-world settings. We provide supplementary material online and will open source our trained model in due course. https://sites.google.com/view/covis-net **Jan Blumenkamp,Steven Morad,Jennifer Gielis,Amanda Prorok CoRL2024**

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


