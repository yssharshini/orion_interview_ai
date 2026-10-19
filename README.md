# Orion - AI-Based Intelligent Interview Training & Performance Evaluation System
VIT SENSE · AI & ML · Project Review 1 → working starter code for Review 2.

## 1. Run it (5 minutes)
```bash
pip install -r requirements.txt        # sentence-transformers / whisper / librosa are the "real AI" parts
sudo apt install ffmpeg                 # (or brew install ffmpeg) needed by Whisper
uvicorn app.main:app --reload           # open http://127.0.0.1:8000
pytest -q tests                         # smoke test: good answer must outscore a bad one
```
Without sentence-transformers the code falls back to TF-IDF so it still runs, but scores are much weaker -
install it for any demo or result you present. `GET /api/health` shows which engines are active.

## 2. Pipeline = your methodology slide
Candidate → website (`frontend/index.html`) → question (`/api/session`) → voice answer → ASR → evaluation → score → report.

| Slide feature | Owner (from your role slide) | File | Method |
|---|---|---|---|
| Voice input & ASR | Rishika | `modules/asr.py` + recorder in frontend | Whisper (`base`/`small`) |
| Semantic accuracy | Harshini | `modules/semantic.py`, `embeddings.py` | Sentence-Transformers cosine vs ideal answer |
| Topic drift | Anuraag | `modules/drift.py` | per-sentence cosine distance to question+ideal |
| Missing concepts + study tips | Srijan | `modules/concepts.py`, `data/question_bank.json` | keyword + embedding match → resource map |
| Backend / API | Sidharth | `app/main.py`, `app/pipeline.py` | FastAPI REST |
| Database | Samyuktha | `app/db.py` | SQLite schema (users, questions, sessions, answers) |
| Frontend & report | Anandita | `frontend/index.html` | recorder, live results, pie chart, study list |
| Fluency & pauses | shared | `modules/fluency.py` | WPM, fillers, long pauses (deterministic) |
| Facial expression (optional) | shared | `modules/facial.py` | MediaPipe landmarks, returns `None` if not installed |
| Final scoring | shared | `modules/scoring.py` | weighted rubric blended with Model 1 |

## 3. The two models you will train ("1 each" - decide who takes which)
Everything else uses pre-existing models (Whisper, Sentence-Transformers, NLI, MediaPipe).

**Model 1 - Interview communication score predictor**  `training/train_communication_score.py`
Predicts a 0-100 human score from the features in `modules/features.py` (semantic, drift, concept coverage, wpm, filler rate, pauses, vocabulary, length, vocal confidence). Ridge vs Gradient Boosting, 5-fold CV, MAE / R² / Spearman.
Needs **human-scored answers**: this is the one thing no public dataset gives you (literature slide, Hickman et al.: label quality drives reliability).
1. Have each teammate record/type ~15 answers to the 12 bank questions (aim ≥300 answers total; add more questions in `question_bank.json`).
2. Two people score each answer 0-100 using the rubric in §5 (average them; report inter-rater agreement).
3. Save as CSV `question_id,transcript,human_score` → `python training/build_features_from_transcripts.py --inp mine.csv` → `python training/train_communication_score.py`.

**Model 2 (optional) - Vocal delivery classifier** (confident / nervous / flat)  `training/train_vocal_delivery.py`
MFCC + pitch + energy → Random Forest, **speaker-independent split**. Trained on RAVDESS and CREMA-D.
`python training/train_vocal_delivery.py --ravdess <dir> --cremad <dir>/AudioWAV`
Caveat to state in the report: acted emotion is only a *proxy* for interview delivery; the emotion→delivery mapping is the `DELIVERY` dict in the script.

## 4. Datasets checklist (your table → where to get them)
| Feature | Dataset | Get it | Used for |
|---|---|---|---|
| Speech transcription | Common Voice, your own recordings | commonvoice.mozilla.org | Whisper WER check only (no training) |
| Semantic accuracy | **your Q&A bank** (`data/question_bank.json`), SQuAD | rajpurkar.github.io/SQuAD-explorer | threshold tuning, experiments |
| Factual consistency | verified Q&A + SNLI/MNLI | nlp.stanford.edu/projects/snli | NLI cross-encoder (optional stretch) |
| Topic drift | answers with relevance annotations | self-made (mark off-topic sentences) | tune `threshold` in `drift.py` |
| Missing concepts | concept labels per question | `question_bank.json` (extend to 40+ questions) | concept coverage |
| Facial expression | FER-2013 | kaggle.com/datasets/msambare/fer2013 | optional classifier in `facial.py` |
| Voice emotion/delivery | RAVDESS, CREMA-D | zenodo.org/record/1188976 · github.com/CheyneyComputerScience/CREMA-D | Model 2 |
| Fluency & pauses | your recordings + transcripts | self-made | deterministic, no training |
| Final scoring | human-scored answers | self-made (see Model 1) | Model 1 + weight tuning |
| Recommendations | concept→resource map | `resources` in `question_bank.json` | rule-based |

## 5. Human scoring rubric (use for Model 1 labels)
Score 0-100 as: Accuracy 35 · Relevance 20 · Completeness 20 · Fluency 15 · Delivery 10 (the same weights as `scoring.WEIGHTS`). 80+ excellent, 65-79 good, 50-64 fair, <50 needs work.

## 6. Evaluation to put in Review 2
- ASR: WER of Whisper on 20 of your own recordings.
- Semantic/drift/concepts: correlation (Spearman) of each module score with human scores.
- Model 1: MAE, R², Spearman on held-out answers vs the pure-rubric baseline (shows whether ML adds value).
- Model 2: per-class F1 + confusion matrix on unseen actors.
- Limitations to acknowledge (your lit survey already does): LLM/ML bias, emotion detection isn't universal, similarity ≠ factual correctness.

## 7. Notes
- `data/scored_answers_SYNTHETIC_demo_only.csv` is generated data to prove the training code runs. **Never report numbers from it.**
- No trained models ship in `models/` on purpose: scoring uses the rubric alone until you train real ones.
- Edit the question bank to add your own questions; each needs `ideal_answer` and `concepts` with keywords.
- Ideas for later: LLM adaptive follow-ups (RAG over resume), AI-assisted-cheating check (paper [6]), React frontend.
