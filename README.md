# Handwritten Digit Recognition

A local Streamlit lab demonstrating Logistic Regression, K-Means, and tabular Q-Learning. The approved scope is in `plans/From_Digits_to_Decisions_Project_Plan.md`.

## Developer setup (Windows / PowerShell)

Prerequisites: Git and Python. Tested on Windows with Python **3.13.7**; other Python versions and platforms have not been verified. No CUDA, API keys, database, or separate ML server is required.

Clone the project and create a virtual environment inside it:

```powershell
git clone https://github.com/Lawrens16/Handwritten-Digit-Recognition-ML.git
cd Handwritten-Digit-Recognition-ML
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501**, or the URL printed by Streamlit. Keep the terminal running; press **Ctrl+C** to stop. These commands use the project environment explicitly, so activation and execution-policy changes are unnecessary. Do not copy someone else's `.venv`; create your own.

For future runs, open PowerShell in the project root and run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Dataset setup: where to get it and where to put it

**No manual dataset download or copying is required to run the app.** Installing the requirements installs scikit-learn, which bundles the Digits dataset. The app calls `load_digits()` to read it locally. After dependency installation, loading the dataset does not require internet access, including on the first run.

The dataset contains **1,797 images**, each **8×8 pixels**, with intensities **0–16** and labels **0–9**. It is not MNIST. Source: [scikit-learn Digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

Verify that it is installed:

```powershell
.\.venv\Scripts\python.exe -c "from sklearn.datasets import load_digits; print(load_digits().images.shape)"
```

Expected output: `(1797, 8, 8)`.

### Optional viewable images

The `dataset/` folder is ignored by Git, so it will **not** appear in a fresh clone. It contains optional visual exports; the app does not load training data from it.

If a teammate shares the exported folder, paste the **whole `dataset` folder directly into the project root**, beside `app.py`:

```text
Handwritten-Digit-Recognition-ML/
    app.py
    requirements.txt
    dataset/
        0/              # Images labeled 0
        1/              # Images labeled 1
        ...
        9/              # Images labeled 9
        index.csv       # Dataset index, label, image path
        pixels.csv      # Original pixel intensities
        overview.png    # Gallery
        README.md
```

Alternatively, generate your own PNGs and index with this entire PowerShell block, run from the project root after installing requirements. It creates/replaces the generated PNGs and `index.csv`:

```powershell
@'
from pathlib import Path
import csv
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

root = Path("dataset")
digits = load_digits()
root.mkdir(exist_ok=True)
with (root / "index.csv").open("w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(["dataset_index", "actual_digit", "image_path"])
    for index, (pixels, label) in enumerate(zip(digits.images, digits.target)):
        (root / str(int(label))).mkdir(exist_ok=True)
        relative = f"{int(label)}/digit_{index:04d}.png"
        grayscale = np.rint(pixels * 255 / 16).astype(np.uint8)
        image = Image.fromarray(grayscale)
        image.resize((256, 256), Image.Resampling.NEAREST).save(root / relative)
        writer.writerow([index, int(label), relative])
print("Exported 1,797 PNG images and dataset/index.csv")
'@ | .\.venv\Scripts\python.exe -
```

Pillow (`PIL`) is already installed through Streamlit. This minimal export creates the ten image folders and `index.csv`; the gallery, original-pixel CSV, and export README are extras in the full shared export.

The PNGs are enlarged for viewing; no image detail is added. `digit_0042.png` means **original dataset index 42**, not held-out sample index 42. The app inspector shows the original dataset index for matching.

Changing or deleting these exported images does not affect training. Pasting MNIST, paper photos, or arbitrary images into this folder does not make the app recognize them; supporting those inputs requires preprocessing and training changes.

## Troubleshooting

- **Python is not recognized:** install Python 3.13 with its PATH option enabled, then reopen PowerShell.
- **No module named streamlit:** use the `.venv` Python executable for both installation and running, as shown above.
- **Cannot find app.py or requirements.txt:** change into the cloned project root first.
- **Port 8501 is busy:** append `--server.port 8502` to the run command and open the printed URL.
- **No dataset folder after cloning:** this is expected. The app uses bundled data; the optional export above creates viewable PNGs.

## Using the application

Each tab has one training button. Before training, it explains what to do. Clicking the button runs learning and replaces that tab's prior result. Selecting samples or clusters retains all trained results in the current session. A new browser session/server restart starts fresh; no models are saved to disk.

- **Recognize:** normalize pixels by 16, stratify an 80/20 split with seed 42, fit Logistic Regression only on training rows, inspect a test image, prediction, actual label, and confusion matrix.
- **Group:** K-Means with ten centers, ten initializations and seed 42 receives normalized vectors only. Inspect centers, counts, and up to ten members. Cluster identifiers are not digit predictions. Silhouette is not accuracy.
- **Navigate:** the fixed deterministic 4×4 FrozenLake map is a separate task. Q-Learning uses alpha 0.1, gamma 0.95, 10,000 episodes, and a 100-step cap. Epsilon starts at 1 and decays by 0.999 to a minimum of 0.05. True terminal updates omit future values; time-limit updates bootstrap. Evaluation freezes the greedy policy for 100 episodes. The route comes from actual environment transitions; loops and failures are reported.

Below the confusion matrix, select an actual digit (row) and predicted digit (column) to list the matching held-out samples and original dataset indices. Select a matching sample to view its image. Off-diagonal cells are mistakes; diagonal cells are correct predictions. Empty cells show a message. These controls reuse the trained model.

The 1,797 images are bundled with scikit-learn in `.venv/Lib/site-packages/sklearn/datasets/data/digits.csv.gz`. `load_digits()` reads this compressed numeric dataset; there are no separate image files to download.

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
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app.py supervised.py unsupervised.py reinforcement.py
.\.venv\Scripts\python.exe -m pip check
```

Six tests cover independently refitting the classifier on training rows (leakage), test normalization/metrics, actual nearest-center membership and counts, terminal/nonterminal updates, successful real navigation, unchanged evaluation table, loop reporting, training/selection through Streamlit AppTest, and confusion-cell filtering with empty-cell handling without retraining. No existing lint, formatter, or security tool configuration was supplied. Dependencies are pinned to the versions installed and exercised here.

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

## Browse the exported dataset
Open dataset/overview.png for a gallery, or dataset/0 through dataset/9 for individual images. All 1,797 PNGs are enlarged for viewing; dataset/index.csv maps filenames to original dataset indices, and dataset/pixels.csv preserves the original 64 pixel values. See dataset/README.md. The app continues to load the bundled data; exported images do not alter training.
