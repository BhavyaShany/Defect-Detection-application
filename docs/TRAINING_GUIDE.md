# Training guide: casting photos → working classifier

This trains the EfficientNetB0 classifier on **2-D photographs** (e.g. the Kaggle *casting product*
dataset) and plugs it into the app as an *image-mode* model. Nothing here was run during development
(no TensorFlow/GPU/data was available), so treat the first run as a check of your setup; use `--smoke-test` first.

## 0. What you are training – and what it will and won't do
* **Input:** one photograph → **output:** a class (e.g. `def_front` / `ok_front`) + confidence. It classifies the *whole image*; it does not locate the defect.
* The Kaggle set is photographs of **one part type** (submersible pump impellers, top view) under controlled lighting.
  A model trained on it recognises *that setup*. It will not generalise to other parts, cameras or lighting without new data. For a customer, expect to collect and label
  **their** images and re-train/validate (see §8).
* Photos and 3D scans are different domains. An image-mode model classifies photos only; 3D scans uploaded while it is loaded are still preprocessed and viewable but stay `UNCLASSIFIED`.

## 1. Get the data
Kaggle → dataset *"Real-life industrial dataset of casting product"* (slug `ravirajsinh45/real-life-industrial-dataset-of-casting-product` – search by title if it moved). Roughly 7,000 grayscale 300×300 images, ≈ 3.7 k defective / 2.9 k ok in `train`, ≈ 0.7 k in `test`.
(The script prints the real counts – trust those.) Unzip so you have:

```
casting_data/
  casting_data/            # (the wrapper folder is detected automatically)
    train/def_front/*.jpeg   train/ok_front/*.jpeg
    test/def_front/*.jpeg    test/ok_front/*.jpeg
```
Roboflow exports in *detection* formats (YOLO/COCO boxes) are not classification folders; use only exports organised as one folder per class.
Check each dataset's licence before using it in a commercial product.

## 2. Environment
Best: **Google Colab with a GPU** (Runtime → Change runtime type → GPU) – use `ml/train_colab.ipynb`, which does steps 2-6 for you.
Local: Python 3.10–3.12, `pip install -r requirements.txt tensorflow`. On CPU only, training can take hours; use fewer epochs for a trial.

## 3. Smoke test (≈ 1–3 min, checks your setup, model unusable)
```bash
python -m ml.train_model --data-dir casting_data --pass-classes ok_front --input-mode image --smoke-test --weights none --out-dir /tmp/smoke
```
Expect: layout `official`, the class list `['def_front', 'ok_front']`, image counts, no `LEAKAGE` error, and a report in `/tmp/smoke`.

## 4. Real training
```bash
python -m ml.train_model --data-dir casting_data --pass-classes ok_front --input-mode image --model-name casting-photo-v1
```
Stage 1 trains the new head on the frozen ImageNet backbone (≤ 15 epochs, early stopping); stage 2 fine-tunes the top 30 layers at a low learning rate (≤ 10 epochs).
`--weights imagenet` downloads the pre-trained weights once (needs internet). Useful flags: `--epochs`, `--fine-tune-epochs`, `--batch-size 16` (if memory is short), `--seed`.

What protects the result from being flattering:
* the **official `test` folder is never used** for training or model selection;
* the script **aborts if any test image is byte-identical to a training/validation image**;
* class weights compensate for the defective/ok imbalance;
* the evaluation is written next to the model (`models/evaluation_report.md/.json`).

## 5. Read the report
* **Recall of the defective class** – of all truly defective parts, how many were caught. This is the number that matters for quality control; missed defects ship to the customer.
* **Confusion matrix** – rows = truth, columns = prediction.
* **Threshold analysis** – for each `CONFIDENCE_THRESHOLD`, the share of parts needing manual `REVIEW`, plus *missed defects* (auto-PASSed but actually defective) and *false rejects*. Pick the threshold from this table, not by guessing; the "indicative threshold" is the smallest with zero missed defects on the test set (small test sets make this optimistic).
* **Cautions** section – automatic warnings (few test images, low defect recall, …).

A sensible bar before showing this to a customer: defect recall ≥ 99 % *at the chosen threshold on data that resembles their production line*, with the review rate they can staff. The public dataset can reach high accuracy because it is clean and consistent – do not quote its numbers as performance on a customer's parts.

## 6. Install the model in the app
Copy these into the project's `models/` folder (next to `README.md`):
`defect_model.keras`, `labels.json` (and keep `evaluation_report.*`). Install on the server the **same TensorFlow version** reported in `evaluation_report.json → environment.tensorflow`
(Keras files are not always portable across major versions). Set `CONFIDENCE_THRESHOLD` in `.env` from the threshold table, then restart the backend:

```bash
pip install tensorflow
uvicorn backend.app:app --port 8000
```
`http://localhost:8000/` should show `"model_loaded": true`, and the dashboard header shows the model name. The upload button now accepts `.jpg .jpeg .png .bmp` as well as 3D formats.

## 7. Try it
Upload a few images **from the `test` folder** (never seen in training): defective ones should come back `FAIL` / `def_front`, good ones `PASS` / `ok_front`, borderline ones `REVIEW`. Compare against the report; a big gap means the app's preprocessing differs from training – tell me and send an example.
Training and serving use the same resize (bilinear, 0–255 RGB, grayscale replicated to 3 channels), so results through the app should match the report closely.

## 8. Towards a sellable product
1. **Per-customer data:** collect images of the customer's actual parts/defects with their QC staff labelling; hold out a test set that is never trained on.
2. **Capture conditions:** fix camera, lighting and part pose; most "model failures" in industrial vision are changes in these.
3. **Validate on-site** (shadow mode: the system classifies, humans still decide) and keep a record of misses.
4. **Monitor drift** (review rate, confidence distribution) and schedule re-training; version each model (the app stores model name + fingerprint with every inspection).
5. **Localisation / 3D:** if customers need *where* the defect is, that requires a detection/segmentation model and annotated regions; for 3D, labelled 3D scans and `ml/build_depth_dataset.py`.
