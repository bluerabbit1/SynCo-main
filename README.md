<<<<<<< HEAD
# SynCo-main
=======
SynCo-OOD: Synthetic-Contrastive Learning for Graph Out-of-Distribution
Detection

Abstract

Out-of-distribution (OOD) detection is crucial for deploying Graph Neural Networks (GNNs) in safety-critical applications. However, the predominant paradigm faces two major limitations. First, OOD detection scores derived from models optimized solely for the in-distribution (ID) task lead to overconfident predictions on OOD data. Second, mitigating this overconfidence often requires auxiliary regularization with labeled OOD samples, creating a critical inconsistency between the training setup and the practical inference scenario where such data is unavailable. To address these challenges, we propose \textbf{SynCo-OOD} (\textbf{Sy}nthetic-\textbf{Co}ntrastive Learning for Graph \textbf{O}ut-\textbf{o}f-\textbf{D}istribution Detection), a novel generative-contrastive framework. First, we propose a contrastive learning objective that compacts ID node representations around a central prototype in the latent space. This process naturally creates a well-defined ID region, enabling a simple yet effective OOD score based on cosine similarity to this prototype, thereby mitigating overconfidence.
Second, to eliminate the reliance on labeled OOD samples, SynCo-OOD uniquely synthesizes pseudo-OOD representations using only ID data and explicitly repels them from the prototype during training.
This dual objective enables the model to learn a discriminative boundary between ID and potential OOD patterns without any exposure to real OOD samples. SynCo-OOD achieves state-of-the-art or competitive performance on five benchmark datasets, despite requiring no OOD exposure during training and offering significant computational efficiency.

Requirements

numpy==1.23.1
ogb==1.3.3
scikit_learn==1.1.1
scipy==1.8.1
torch==1.9.0
torch_geometric==2.0.3
torch_sparse==0.6.12


Train Model
Run main.py to train with the default setting, and the datasets will be downloaded automatically into ./data/.
You can also customize the settings in config_files
>>>>>>> fb5969d (first commit)
