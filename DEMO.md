# Demo script (about five minutes)

1. Start `python -m streamlit run app.py`. Explain that the digit dataset is built in and all learning runs locally.
2. Recognize: click Train classifier. Explain that `fit` sees only training labels. Show test accuracy and the confusion matrix (rows actual, columns predicted). Select two held-out images. The actual label is shown beside the model prediction; acknowledge any mistake.
3. Group: click Train clusters. Show ten learned center images and choose a cluster. Explain that these IDs are arbitrary and can mix digits. Member counts total 1,797. Silhouette measures cohesion/separation, not digit accuracy.
4. Navigate: click Train agent. Explain 16 states, four actions, reward +1 only at the goal, and zero otherwise. Show the rolling training rate, route, and Q-table. Evaluation has no exploration or updates and repeats the same deterministic map.
5. Contrast learning from labels, similarity without labels, and rewards through interaction. Explain that training occurs only on button clicks; selecting samples uses the existing model.

## Q&A cues

- Logistic Regression learns class weights from labeled training examples; it is not a neural network.
- The held-out split estimates performance on unseen dataset examples. No claim is made about arbitrary photos or handwriting uploads.
- K-Means assigns nearest centers then recomputes means. Cluster 3 is not digit 3.
- Q-learning estimates action values. Exploration tries alternatives; exploitation chooses the highest learned value.
- Terminal targets contain only the immediate reward; time-limit truncation still bootstraps.
- Evaluation repeats one map/start, so 100/100 does not establish generalization.
- CPU is sufficient for 1,797 small images and a 16-by-4 table. CUDA complexity adds little value here.
- Tests check data separation, memberships, updates, frozen evaluation, and UI state. The team must review and understand AI-assisted code and acknowledge assistance under instructor rules.

Before presenting: add team names to the slides, rehearse on the submission laptop, and capture screenshots of actual results as a backup. Confirm the instructor's submission channel and cutoff. Submission itself is a team action.
