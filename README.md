# Awesome_AI_for_Robotics
As ChatGPT and RT2 appears, many researchers start to do reseaches about AI for robotics. Many scholars have different names for it, including embodied AI, smart robotics, etc. I don't care what its name really is, I just want to share what I have learnt. Hope it helps!
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



## Papers

### (1) Reinforcement Learning & Imitation Learning
1.[HIQL: Offline Goal-Conditioned RL
with Latent States as Actions](https://proceedings.neurips.cc/paper_files/paper/2023/file/6d7c4a0727e089ed6cdd3151cbe8d8ba-Paper-Conference.pdf)
This paper introduces a new way to improve the performance of action-free offline reinforcement learning. Most of the time, when we want to do training in action-free offline reinforcement learning, we need to train a value network and use it to update a Q function network. However, the authors in this paper have used 2 Q function networks. One of them is used to predict a high level policy, and the other is used to learn a low level policy. **Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine. Arxiv 2024** 
![image](https://github.com/user-attachments/assets/0f2547b4-4fbb-43a3-9f15-ff863d0e3a05)


2.[Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control](https://arxiv.org/pdf/2401.16889) This paper introduces a multi-stage method for bipedal locomotion control. The network inputs the short term history and the CNN feature of 2 second long history to output the policy. The method has 3 stages. First, it trains a single task by RL. Second, it trains tasks randomly to learn a general policy. Finally, it randomize the dynamics (the parameters of the environments and noise) to bridge the sim2real gap.  **Zhongyu Li, Xue Bin Peng, Pieter Abbeel, Sergey Levine, Glen Berseth, Koushil Sreenath. IJRR 2024** 

3.[Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance](https://openreview.net/pdf?id=6FGlpzC9Po)
This paper learns a value action conditioned on the language guide based on lanrge dataset training. When in reference, we can use the value function the sample the best action in multiple choices. The formula below is how to train the value function Q. **Mitsuhiko Nakamoto, Oier Mees, Aviral Kumar, Sergey Levine. CoRL 2024**
![image](https://github.com/user-attachments/assets/62870f16-21aa-4909-a35e-d07bad3695b7)

4.[Reconciling Reality through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation](https://arxiv.org/pdf/2403.03949)
A new paradigm for learning robot policy. The authors first use imitation learning to learn the real world policy, reconstructs the environment and uses reinforcement learning to learn the policy in the simulator, and transfer the policy to the real world using teacher-student model (because sim and real have different inputs). I admit that there is performance increase, but I doubt whether it is applicable to the real world particularly in large complicated scenarios.  But we can learn from the paper that, we can add some regularization when learning the strategies so we can adopt the advantages of the previous policy. **Marcel Torne, Anthony Simeonov, Zechu Li, April Chan, Tao Chen, Abhishek Gupta2, Pulkit Agrawal. Arxiv 2024** 

5. [OGBENCH: BENCHMARKING OFFLINE GOAL-CONDITIONED RL](https://arxiv.org/abs/2410.20092)
A benchmark for offline goal-conditioned reinforcement learning to assess the capability of algorithms to deal with challenges in this field. The challenges including (1) Learning from suboptimal, unstructured data (2) Goal stitching (3) Long-horizon reasoning (4) Handling stochasticity, which motivates the design of OGBench. Additionally, OGBench illustrated the open problems for reaearchers to investigate according to the results, which is a foudation for researchers to achieve the goal of offline GCRL:  building foundation models for general-purpose behaviors.
### (2) Vision-Language Action Model 
1.[MiniVLA: A Better VLA with a Smaller Footprint](https://ai.stanford.edu/blog/minivla/)
The OpenVLA by far the sota method in open-source vision-language action models. However, the model has a slow inference and training speed and only supprt per-image input. In this work, the authors use action chunking (multi-action output) and multi-image input to improve the performance. What's more, the authors also use Qwen 0.5B model as the backbone to reduce the size of the backbone mode. **Suneel Belkhale and Dorsa Sadigh. The Stanford AI Lab Blog 2024** 

2.(about to read) action chunking transformer


### (3) Dataset & Dataset Generation
1.[Automated Creation of Digital Cousins for Robust Policy Learning](https://arxiv.org/pdf/2410.07408)
This is a kind of domain randomization method. When training a robot policy based on imgaes input, the authors leverage the depth information and the basic layouts to match different kinds of similar objects to replace the original objects in the scene (The authors defind it as the "digital cousin") and render the new objects to form a new image. When training on different kinds of objects in the same layout, we can bridge the sim2real domain gap. **Tianyuan Dai, Josiah Wong, Yunfan Jiang, Chen Wang, Cem Gokmen, Ruohan Zhang, Jiajun Wu, Li Fei-Fei CoRL 2024** 
![图片](https://github.com/user-attachments/assets/f85388b0-4bf7-4ddf-9874-74e83dc095bc)

### (4) Inference Speed 
1. [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
Speculative Execution: an optimization technique, where a task is performed in parrallel, can accelerate inference without changing the model architecture, withing re-training, without changing the training procedures and without changing the model output distributions. **Yaniv Leviathan, Matan Kalman, Yossi Matias ICML 2023**
![image](https://github.com/user-attachments/assets/41a6eb6b-4e0d-4c24-90c9-eb64f644dfb2)



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

### Nov.2024
1.Bridge Sim2Real Gap: domain randomization, system identification, or improved simulator visuals <br>
2.When connecting to huggingface is not convenient, we can set: **os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'**<br>
3.When "labels" in "CausalLMOutputWithPast" is -100, it means we can neglect it.


