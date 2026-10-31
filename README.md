# Handwritten Digit Recognition

A local Streamlit lab demonstrating Logistic Regression, K-Means, and tabular Q-Learning. The approved scope is in `plans/From_Digits_to_Decisions_Project_Plan.md`.

## Windows setup

Tested on Windows with Python 3.13.7. From the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If activation is restricted, use `.\.venv\Scripts\python.exe` in place of `python`; no execution-policy change is needed. Open the local URL printed by Streamlit. Stop with Ctrl+C. After dependency installation, no data downloads or online services are needed. Do not enable Streamlit's optional sharing/deployment features for the local demo.

## Using the application

Each tab has one training button. Before training, it explains what to do. Clicking the button runs learning and replaces that tab's prior result. Selecting samples or clusters retains all trained results in the current session. A new browser session/server restart starts fresh; no models are saved to disk.

- **Recognize:** normalize pixels by 16, stratify an 80/20 split with seed 42, fit Logistic Regression only on training rows, inspect a test image, prediction, actual label, and confusion matrix.
- **Group:** K-Means with ten centers, ten initializations and seed 42 receives normalized vectors only. Inspect centers, counts, and up to ten members. Cluster identifiers are not digit predictions. Silhouette is not accuracy.
- **Navigate:** the fixed deterministic 4×4 FrozenLake map is a separate task. Q-Learning uses alpha 0.1, gamma 0.95, 10,000 episodes, and a 100-step cap. Epsilon starts at 1 and decays by 0.999 to a minimum of 0.05. True terminal updates omit future values; time-limit updates bootstrap. Evaluation freezes the greedy policy for 100 episodes. The route comes from actual environment transitions; loops and failures are reported.

## Measured results

Observed October 31, 2026, Python 3.13.7; see `results.json`. Timing varies by hardware and process warm-up.

| Algorithm | Actual outcome | CPU elapsed time including metrics/evaluation |
| --- | --- | --- |
| Logistic Regression | 344/360 correct, **95.56%**; 16 errors | 0.07 seconds |
| K-Means | 1,797 members, ten centers; silhouette **0.1824** | 0.16 seconds |
| Q-Learning | **100/100** greedy evaluation wins | 0.80 seconds |

The learned route in this run was 0 → 4 → 8 → 9 → 13 → 14 → 15. These are recorded observations, never hardcoded into training or UI results. CPU already completes all three tasks in about one second; CUDA was omitted to avoid extra frameworks and setup. No CUDA benchmark was run, so no numerical GPU speedup is claimed.

## Code guide

`supervised.train_classifier()` loads and splits the dataset, calls `fit`, then calls `predict` on held-out rows. `unsupervised.train_clusters(features)` alternates nearest-center assignment and mean updates through sklearn K-Means, accepting no labels. `reinforcement.train_agent()` samples real transitions, calls `update_q()`, and records rewards. `evaluate_policy()` never updates Q; `rollout()` derives a capped route. `app.py` holds results in session state and renders the three demonstrations through Matplotlib, avoiding pygame.

## Verification

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q app.py supervised.py unsupervised.py reinforcement.py
python -m pip check
```

Five tests cover independently refitting the classifier on training rows (leakage), test normalization/metrics, actual nearest-center membership and counts, terminal/nonterminal updates, successful real navigation, unchanged evaluation table, loop reporting, and training/selection through Streamlit AppTest. No existing lint, formatter, or security tool configuration was supplied. Dependencies are pinned to the versions installed and exercised here.

## Presentation and limitations

Open `presentation.html` in a browser for eight offline slides. Arrow keys/Page Down advance; Page Up goes back; print produces one slide per page. Add actual team names before presenting. `DEMO.md` gives the demo sequence and Q&A cues. Capture screenshots on the presenting laptop as backup.

The dataset is 1,797 built-in 8×8 digit images, not MNIST, arbitrary photos, or uploaded handwriting. No canvas, OCR, camera, deep learning, online AI service, database, authentication, or hosting is included. Low silhouette and mixed clusters are legitimate observations. The RL success rate repeats the same map/start and does not demonstrate generalization. Team rehearsal, instructor-specific AI acknowledgement, submission channel/cutoff, and actual submission remain human responsibilities.

## Sources

- [Digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- [Termination versus truncation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
- [Streamlit session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)

The supplied plan references the instructor guidelines; that original PDF was not in this checkout. AI assisted implementation and testing; team members should review the code and follow the instructor's attribution rules.
