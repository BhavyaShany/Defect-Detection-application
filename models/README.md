# Model folder

The backend loads two files from here (paths configurable via `MODEL_PATH` / `MODEL_LABELS_PATH`):

| File | Purpose |
|---|---|
| `defect_model.keras` | Trained EfficientNet classifier (Keras file, or a SavedModel directory) |
| `labels.json` | Class names (index = model output index), which classes mean "pass", input mode (`image` for photos; `depth`/`normal`/`depth_normal` for 3D scans), image size |

**No trained model ships with this project.** Until both files exist, the API starts normally,
preprocesses and displays scans, and stores them as `UNCLASSIFIED` -- it never invents a prediction.

Create them with the training tool (see `README.md` > *Training a model*):

```bash
# photographs (Kaggle casting set)
python -m ml.train_model --data-dir casting_data --pass-classes ok_front --input-mode image
# depth maps from labelled 3D scans
python -m ml.train_model --data-dir dataset --pass-classes ok --input-mode depth
```
Full walkthrough: `docs/TRAINING_GUIDE.md`.

`labels.json.example` shows the format. The `input_mode` **must match how the training images
were produced** (`ml/build_depth_dataset.py` uses the same preprocessing as the live API).
