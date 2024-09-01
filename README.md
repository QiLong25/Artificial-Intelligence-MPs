# Artificial-Intelligence-MPs
Artificial Intelligence-UIUC-CS course assignments

## MP1: Probability
 *  **Joint, Conditional, and Marginal Distributions**.

 *  **Mean Vector and Covariance Matrix**.

 *  **Expected Value of a Function of an RV**.

## MP2: Naive Bayes
*Movie Review Classification*

[Stanford Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

 *  **Learning a Naive Bayes Model**: Maximum Likelihood, Stop Words, Laplace Smoothing.

 *  **Decisions Using a Naive Bayes Model**.

 *  **Optimizing Hyperparameters**.

## MP3: Hidden Markov Model
*Word Sequence*

 *  **Baseline Tagger**.

 *  **Viterbi**: HMM Tagger.

 *  **Viterbi_ec**: Improve HMM Tagger by applying emission smoothing to match the real probabilities for unseen words.

## MP4: Perceptron
 *  **Single Classical Perceptron Training & Testing**.

## MP5: Neural Networks
 *  **PyTorch Tutorial**: Datasets, Dataloaders, Tensors.

 *  **Neural Net Layers**: linear layers, activation functions.

 *  **Model Training**: forward(), loss function, back propagation, optimizer (SGD).

 *  **Convolutional Neural Network**: convolution, pooling, flatten.

 *  **L2 Regularization**: Adam, Adagrad, weight_decay.

 *  **Hyperparameters**: epochs, batch_size, learning_rate, learning_rate_decay.

## MP6: Search
*Find path in maze.*

 *  **BFS**.

 *  **A\***: Estimate cost from current space to waypoint, Faster BFS.

![BFS](https://github.com/QiLong25/Artificial-Intelligence-MPs/assets/143149589/fe13ceae-0772-4abe-859d-2fbe94ccc2a8)

 *  **Multi-waypoints**: Search path through all waypoints.

![multi-waypoints](https://github.com/QiLong25/Artificial-Intelligence-MPs/assets/143149589/f5951b6d-37ef-4a6e-aebf-f02ef21a121b)

## MP7: Minimax
*Chess AI-agent*

https://github.com/QiLong25/Artificial-Intelligence-MPs/assets/143149589/f2691704-43df-4d2b-8133-513b264e24f0

 *  **Minimax Search**: Optimize action based on minimax tree search of various depth.

 *  **Alphabeta Search**: Alphabeta pruning on Minimax method, reduce evaluations on unnecessary leaf nodes.

 *  **Learn Heuristic**: train a neural network to compute a better heuristic.

## MP8: Repeated Games
*The Lunch Game*

 *  **Episodic Games: Gradient Ascent**, find optimal policy leading to Nash Equilibrium. (*Problem: cannot converge*)

![orbit_plot](https://github.com/QiLong25/Artificial-Intelligence-MPs/assets/143149589/270c4a08-5cf7-4ea4-a426-08e6f0c80b0c)

 *  **Episodic Games: Corrected Ascent**, symplectic correction making use of Hessian.

![hessian_plot](https://github.com/QiLong25/Artificial-Intelligence-MPs/assets/143149589/07c46bdb-82d8-45a1-94e8-3e578fe80270)

 *  **Sequential Games**: Optimize policy based on its own and opponent's previous behavior.

## MP9: Transformers
*Build Transformer from Scratch*

 *  **Multi-head Attention**: Q, K, V matrix multiplication with weights, Scaled Attention Mechanism.

 *  **Positional Encoding**: sine and cosine of different frequencies.

 *  **Mask**: padding mask (pad to same length sequences), attention mask (avoid decoder looking forward).

 *  **Special Tokens**: <sos>, <eos>, used in decoder.

 *  **Encoder**: Stack of encoder layers, each layer has two sets of sub-layer operations (multi-head **self-attention** + position-wise, fully connected feed-forward network), dropout, residual, layer normalization.

 *  **Teacher-forcing Decoder**: Stack of decoder layers, each layer has three sets of sub-layer operations (multi-head **self-attention** + multi-head **encoder-decoder attention** + position-wise, fully connected feed-forward network), dropout, residual, layer normalization.

 *  **Auto-regressive Decoder**: Decoder one-step-forward, layer cache, output append to input.

![image](https://github.com/QiLong25/Artificial-Intelligence-MPs/assets/143149589/41b68db1-6b5e-46f3-a681-e81815a00dce)

## MP10: Markov Decision Process
*Grid World*

![grid_world](https://github.com/QiLong25/Artificial-Intelligence-MPs/assets/143149589/d660547f-5c05-46cc-ae75-01f3d71b651d)

 *  **Transition Matrix**: probability distribution of state transition over all state-action pairs.

 *  **Value Iteration**: one-step greedy Bellman update for each iteration till stable.

 *  **Policy Evaluation**: one-step Bellman approximation for each iteration till stable.

## MP11: Policy Gradient
 *OpenAI Gym* 

 *  **Vanilla Policy Gradient**: Implement Policy Gradient Loss, Train a neural network to return optimal action, Policy Training.

 *  **Advantage Estimation**: Train another neural network to stimate the mean future return, use difference to long-term return as advantage.

 *  **PPO**: Monitor policy update step-size, clip too large shift from original policy.

https://github.com/QiLong25/Artificial-Intelligence-MPs/assets/143149589/583d3bde-7755-4ebc-8f18-c08186afa7e9

## Disclaimer

THIS PROJECT IS INTENDED FOR WORK DISPLAY AND SHARING PURPOSES ONLY. THE CODE AND MATERIALS PROVIDED HERE SHOULD NOT BE SUBMITTED AS YOUR OWN WORK IN ANY FORM. USE OF THIS CODE FOR ACADEMIC ASSIGNMENTS OR EXAMS WITHOUT PROPER ATTRIBUTION OR PERMISSION MAY CONSTITUTE ACADEMIC MISCONDUCT INCLUDING PLAGIARISM. THE AUTHOR OF THIS PROJECT IS NOT RESPONSIBLE FOR ANY CONSEQUENCES ARISING FROM THE IMPROPER USE OF THIS CODE.



















