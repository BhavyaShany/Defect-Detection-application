# Presentation guide

## The recommended setup (decided for you)
1. **Primary:** the app **deployed on a cloud VM** (Deployment guide, Track 2) with HTTPS and the demo password. This is what your professor asked to see: a URL that works from any browser.
2. **Backup 1:** the same app on your laptop in single-process mode (Track 0, `http://localhost:8000`). Same data, same screens, no internet needed.
3. **Backup 2:** a screen recording (2-3 min) of the full demo flow. Record it the day before.
4. **For the company:** the *same live demo*, framed as a **pilot-ready prototype**, plus a slide on how it would run on their conveyor (below). Do not hand them the public URL.

Do the rehearsal with Track 0 first, then deploy, then rehearse again on the deployed URL. Do not deploy for the first time on demo day.

## Demo script (about 8 minutes)
| Min | Show | Say |
|---|---|---|
| 0-1 | Problem slide | Manual inspection is slow and inconsistent; we classify cast-part photos automatically |
| 1-2 | Architecture picture (README) | Upload/camera → preprocessing → EfficientNet classifier → decision → dashboard with live updates |
| 2-5 | **Live:** upload 2 defective photos (FAIL), 2 good photos (PASS), 1 borderline (REVIEW if you have one) | Class, confidence, per-stage processing time (measured), history updates live |
| 5-6 | **Live:** upload a selfie / random photo | The input check refuses it instead of giving a confident wrong answer |
| 6-7 | Evaluation slide from `evaluation_report.md` | 99.2 % accuracy on 651 held-out images; recall for defects 99.1 %; confusion matrix; how the threshold was chosen |
| 7-8 | Limitations + next steps | See the table below |

**Prepare before you start:** a folder with the exact images you will upload (named in order), the model warmed up (upload one image before the audience arrives), browser zoom set so the result panel is readable, other tabs closed, phone hotspot as a network backup.

## What you can and cannot claim

| You can say | You must not say |
|---|---|
| "On 651 held-out test photos, with exact duplicates removed, the model reached 99.2 % accuracy and caught 449 of 453 defects" | "The system is 99 % accurate" (it is accuracy on one public dataset, one part type, one camera setup) |
| "Low-confidence parts go to manual review; non-casting images are refused" | "It never misses a defect" (it missed 4 of 453 in the test; those were confident errors) |
| "The classifier says whether a part is defective; it does not locate the defect" | "It highlights the defect" (the 3D colour map is geometry only, and photos have no localisation) |
| "Measured processing time is about X ms per image on this server" (read it from the dashboard) | "Real-time" without giving the number, or any number you did not measure |
| "Designed to run on a conveyor line; a pilot would retrain on the customer's own images" | "Ready for production" |

## For the company: how it would work on their line
* **Edge agent (to be built):** a service next to the camera grabs a frame on a trigger, runs the same model, and signals a diverter or PLC. **Anything not clearly PASS (REVIEW, refused, error) goes to manual inspection.** The line never depends on the internet.
* **Cloud dashboard (what you demo):** receives results for history, statistics and export.
* **Pilot proposal:** (1) they supply a few hundred to a few thousand labelled photos of their own good and defective parts, taken with the production camera and lighting; (2) you retrain and report accuracy, missed defects and review rate on a held-out set; (3) run in *shadow mode* (system classifies, people still decide) for a few weeks; (4) agree the acceptable miss rate **before** switching on automatic rejection.
* **Honest status to state:** works on the public photo dataset; not yet validated on their parts; no user accounts, audit trail or integration with their MES/PLC yet.

## Likely questions
| Question | Honest answer |
|---|---|
| Will it work on our castings? | Unknown until trained and tested on your images; the public-dataset score does not transfer |
| How fast is it? | Show the measured ms per image from the dashboard; depends on server hardware; a GPU or an optimised model would be faster |
| What about 3D scans? | The app loads and visualises 3D scans, but classifying them needs labelled 3D scans we do not have yet |
| Where is our data stored? | In the deployment's own database volume; accounts, retention and backups are the next work item |
| What if the camera or lighting changes? | Accuracy can drop; that is why the pilot validates the exact setup and we monitor the review rate |
