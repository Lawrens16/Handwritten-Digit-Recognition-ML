# Handwritten Digit Recognition

## 1. Goal and boundaries

Build one local Python application demonstrating supervised learning, unsupervised learning, and reinforcement learning. Optimize for a small, working, explainable final lab project, rather than a production system. Team completion and submission target: November 2, 2026 (Philippine time). The supplied guidelines list November 10, 2026 as the presentation date; keep that date unless the instructor changes it. Submit the completed project on November 2, not just this plan.

The assignment requires a working implementation of at least one algorithm per paradigm, a live demonstration, and slides explaining algorithm choice and code execution. Its rubric is algorithmic accuracy (30%), execution and usability (30%), conceptual presentation (25%), and collaboration/Q&A (15%). Three separate demonstrations within one application are a reasonable interpretation of the supplied guidelines; the guidelines do not explicitly require a shared problem.

Deliver: source code, tested dependency versions, Windows setup instructions, a short demo script, and approximately eight slides. This document is the implementation plan, not a completed or tested application.

Use the RTX 3050 laptop with 16 GB RAM; the RTX 4060 laptop with 32 GB RAM is also suitable. Run these small algorithms on the CPU. No CUDA, deep learning framework, online AI API, database, authentication, hosting, or GPU configuration is necessary. AI assists development; the finished application runs its own ML algorithms locally.

## 2. Minimal feature set

| Tab | Algorithm | Input | Required output |
| --- | --- | --- | --- |
| Recognize | Logistic Regression | A selected held-out 8x8 digit image | Predicted digit, actual digit, test accuracy, confusion matrix |
| Group | K-Means, k=10 | Unlabeled pixel vectors from the same dataset | Ten cluster-center images, example members, cluster counts, silhouette score |
| Navigate | Tabular Q-Learning | Fixed deterministic 4x4 FrozenLake map | Training progress, learned path, Q-table, evaluation success rate |

One train button per tab, a seed fixed to 42, and short plain-language explanations are enough. Keep trained results in Streamlit session state so changing a sample does not retrain a model. Training buttons must actually execute learning. Model persistence on disk is optional, not required for this small dataset.

Explicitly excluded: handwriting canvas, arbitrary image upload, camera capture, OCR for words/pages, A-Z, multiple competing models, extensive parameter tuning, user accounts, remote deployment, and complicated animations. Do not add these during implementation.

## 3. Dataset and supervised learning

Use sklearn.datasets.load_digits(), not MNIST. The built-in dataset contains 1,797 images, each 8x8 pixels, with 64 features and intensity values from 0 to 16. Its labels are the digits 0-9. It avoids a separate dataset download.

1. Load images, flattened feature vectors, and labels.
2. Divide pixel values by 16.0. This fixed normalization does not learn from test data.
3. Split 80% training / 20% testing using stratify=labels and random_state=42.
4. Fit LogisticRegression(max_iter=2000) using training samples and labels only. Investigate and resolve any convergence warning rather than suppressing it.
5. Predict on the untouched test split; compute accuracy and a confusion matrix ordered 0-9.
6. Allow the presenter to select any held-out sample by index, display the enlarged image, predict its digit, and reveal the actual label.

Explanation: the model learns a weighted score for each digit from labeled examples and chooses the highest-probability class. Explain fit(), predict(), and the role of the train/test split. Do not describe Logistic Regression as a neural network.

Success criterion: correct data separation, real predictions, and honest held-out results. An accuracy of 90% or above is a practical development target, not an assignment requirement or promised result. Do not repeatedly tune against the test set; if tuning becomes necessary, use a validation split from the training set.

## 4. Unsupervised learning

1. Independently use the dataset's normalized pixel vectors without passing its digit labels into clustering.
2. Fit KMeans(n_clusters=10, n_init=10, random_state=42).
3. Reshape each 64-value centroid into an 8x8 image and display the ten centers.
4. Let the presenter choose a cluster and view up to ten actual member images and its member count.
5. Calculate silhouette score in the same normalized feature space used for clustering.

The ten clusters are a chosen demonstration setting because the dataset contains ten digit classes. Their identifiers are arbitrary: Cluster 3 does not mean digit 3, clusters may mix digits, and multiple clusters may represent different writing styles of the same digit. No labels are needed for silhouette score. Do not report cluster IDs as predicted digit labels or claim that silhouette is classification accuracy. A scatter plot or dimensionality reduction is unnecessary for this minimum version.

Explanation: K-Means alternates between assigning each image vector to the nearest center and updating centers to their assigned samples' means.

Success criterion: reproducible clustering without target labels, accurate counts, understandable centers/member images, and an honestly reported score. There is no required silhouette threshold.

## 5. Reinforcement learning

Use Gymnasium FrozenLake-v1 with map_name='4x4' and is_slippery=False. This supplies a small existing environment rather than requiring a custom simulator. The maze is a separate learning task, not a digit recognition component.

State: one of 16 cells. Actions: left, down, right, up. Reward: +1 for reaching the goal and 0 otherwise, following the environment. Holes end an episode unsuccessfully. Keep the map visible with start, holes, goal, and learned route clearly marked.

Starting settings, to be tested during development:

- Zero-initialized Q-table with shape 16x4.
- Learning rate alpha=0.1; discount gamma=0.95.
- Epsilon-greedy exploration: epsilon starts at 1.0, decays by 0.999 each episode, and has a floor of 0.05.
- Up to 10,000 training episodes; at most 100 steps per episode.
- Seed NumPy's generator and the environment reproducibly. Reset the environment between episodes without resetting the random generator to the same state every time.
- Break on termination or truncation. For an actual terminal transition, the update target is the immediate reward; otherwise include gamma times the maximum next-state Q-value.

Q update: Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)].

Train through actual environment interactions. Show rolling goal-reaching rate over 100 training episodes. After training, evaluate a frozen greedy policy for 100 episodes with exploration disabled and no Q-table updates. Show a capped rollout of the learned policy and its Q-table. Do not hardcode a successful route. Report a loop or failure if the learned policy fails to reach the goal.

Success criterion: training changes the Q-table and the evaluated policy reaches the goal on the fixed deterministic map. A reliable 100/100 evaluation result is the development target for this simple map, not a generalization claim. Evaluation repeats the same map and start; explain that limitation. If training fails, inspect exploration and terminal handling before changing settings; never fabricate results.

## 6. Small code structure and dependencies

- app.py: Streamlit tabs, controls, session state, displays.
- supervised.py: dataset split, training, predictions, metrics.
- unsupervised.py: clustering and cluster summaries.
- reinforcement.py: environment, Q-learning, evaluation, rollout.
- tests/test_learning.py: a few meaningful checks for leakage, clustering label independence, and learned navigation.
- requirements.txt: only direct dependencies, pinned to versions verified on the development laptop.
- README.md: installation, run command, concepts, actual results, limitations, demo instructions, and sources.

Use Python, Streamlit, scikit-learn, NumPy, Matplotlib, and Gymnasium. Avoid pygame by rendering the grid through Matplotlib rather than Gymnasium's graphical renderer. Python 3.11 or 3.12 is a reasonable starting environment; verify dependency compatibility during installation. Do not invent current package versions.

Windows setup pattern: create .venv, activate it, run python -m pip install -r requirements.txt, and launch with python -m streamlit run app.py. Include the exact tested commands in README. After initial dependency installation, the application should work offline using built-in data and the local environment.

## 7. AI-assisted build sequence

Work in small completed stages rather than asking an agent to generate an elaborate system in one pass. Have AI explain each algorithm and important function after implementation. Humans run the software and verify the explanation against the code.

1. Scaffold: requirements, README, three tabs, module stubs. Confirm the app opens.
2. Supervised: implement the split and classifier, verify held-out results, then connect the UI.
3. Unsupervised: implement clustering with no labels, verify counts/centers, then connect the UI.
4. Reinforcement: implement and validate training/evaluation without the UI first, then add the grid and progress display.
5. Integrate: preserve session state, label all results clearly, check offline operation and fresh-start behavior.
6. Prepare presentation using measured results and screenshots of the actual application. Rehearse Q&A and the live demo.

Suggested initial prompt for the coding agent:

> Build a minimal local Python/Streamlit final lab project titled "Handwritten Digit Recognition." Follow this plan. Use sklearn's built-in 8x8 Digits dataset, Logistic Regression with a stratified 80/20 split, K-Means on unlabeled pixel vectors, and real tabular Q-Learning on deterministic 4x4 Gymnasium FrozenLake. Run on CPU. Implement one stage at a time and verify it before moving on. Keep code small, use descriptive names, avoid duplicated logic, and include only meaningful correctness checks. Show actual measured results; never hardcode predictions, metrics, or successful paths. Keep training out of normal UI reruns. Provide Windows setup commands, a concise README, and explanations we can defend in Q&A. Do not add a drawing canvas, image uploads, accounts, database, deployment, extra models, or AI API calls. Begin with the scaffold, then supervised, clustering, reinforcement, and integration.

Use follow-up prompts tied to one failing behavior or stage. Ask for the reason behind fixes and rerun the relevant check. Review AI output before accepting it. Follow the instructor's rules for AI assistance and acknowledge its use where required.

## 8. Verification and definition of done

- Fresh virtual environment installs successfully using recorded dependency versions.
- Application starts with the documented command and works offline after installation.
- All three tabs work from a fresh session and show helpful instructions before training.
- Classifier sees training data only; sample prediction comes from the test split.
- Confusion-matrix counts total the number of test samples; predictions are 0-9.
- K-Means never receives labels; cluster counts sum to 1,797 and ten centers are shown.
- Q-learning really updates values, handles terminal transitions correctly, and produces a goal-reaching rollout within the step cap for the chosen seed.
- Evaluation leaves the Q-table unchanged and uses no exploration.
- Changing samples or tabs preserves trained results; reset/retrain behavior is predictable.
- README and slides report observed values, including mistakes and limitations.
- Every member can explain all three paradigms and identify the relevant functions.

These checks protect algorithm correctness; do not spend time creating a large test suite for every UI label.

## 9. Schedule and participation

| Date / work block (2026, Philippine time) | Work | Completion check |
| --- | --- | --- |
| October 31, remaining work session | Freeze scope, install environment, create three-tab scaffold; use AI to implement classification and clustering | Classifier shows held-out predictions and metrics; K-Means shows centers and members without training labels |
| November 1, morning | Implement and verify Q-learning before connecting it to the interface | Real Q-table updates, correct terminal handling, frozen-policy evaluation, and successful learned route |
| November 1, afternoon | Integrate all tabs; verify session state, fresh installation, and offline execution on the submission laptop | All three demonstrations work from a fresh session using the README commands |
| November 1, evening | Finish approximately eight slides, README, results, demo script, and group rehearsal | All outputs are actual measured results; each member can explain the algorithms; features are frozen |
| November 2, before submission | Run final smoke check; fix blockers only; prepare submission archive and verify its contents | Source, dependency file, README, slides, and requested supporting files are present; exclude .venv, caches, and secrets |
| November 2 | Submit through the instructor's required channel before the actual cutoff; retain a backup and submission receipt | Upload opens correctly and submission is confirmed |

The exact November 2 cutoff and submission channel have not been supplied. Check the class announcement; aim to submit early enough to handle upload problems. This schedule is a team target, not a claim that the instructor changed the original presentation date. Submission remains a team action; this plan does not submit or schedule anything automatically.

Use AI to complete one small module at a time and immediately run it. If time becomes tight, remove cosmetic styling and animations first. Retain all three paradigms, honest evaluation, the working demo, and the required slides. Do not add features after November 1 evening.

For four members: one leads classification, one clustering, one Q-learning, and one integration/documentation. Everyone reviews another module and speaks during presentation. For smaller groups, combine integration with classification. Group limit in the assignment: ideally up to four, absolute maximum five. AI accelerates implementation but does not replace members understanding the work.

## 10. Slides and live demonstration

Eight slides: title and members; goals and three paradigms; dataset and preprocessing; classifier/code/results; clustering/code/results; Q-learning/state-action-reward/update; integrated demo/results; limitations and conclusion/references.

Demo sequence: open the app locally; train classifier and predict two held-out samples; train K-Means and inspect mixed cluster members; train Q-learning, explain reward progress, and display the learned route; finish by contrasting labeled examples, unlabeled similarity, and reward-driven interaction. Rehearse to fit the instructor's allotted time. Keep screenshots as backup evidence, while still preparing for the required live functional run.

Prepare answers to: Why is Logistic Regression supervised? Why not use classification accuracy for raw cluster IDs? What does the agent learn? How is exploration different from exploitation? Why use a test split? Why is a GPU unnecessary? How did the team verify AI-generated code?

## 11. References

- Supplied Final Project Guidelines.pdf (read October 31, 2026).
- Digits dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- K-Means: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- Silhouette score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- FrozenLake: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- Q-Learning tutorial: https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/

Implementation settings and dates above are proposed team decisions, not additional instructor requirements. Exact performance and dependency versions must be established when the program is built.
