"""Facial expression (optional, Sidharth/Anandita can extend). Rule-based on MediaPipe face landmarks;
returns None if mediapipe/opencv are not installed so the rest of the pipeline is unaffected.
Swap in a FER-2013 CNN classifier here if you want a trained model."""
def analyze_video(video_path, max_frames=150):
    try:
        import cv2, mediapipe as mp
    except Exception:
        return None
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(video_path)
    seen = faces = smiles = 0
    while seen < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        seen += 1
        r = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if r.multi_face_landmarks:
            faces += 1
            lm = r.multi_face_landmarks[0].landmark
            width = abs(lm[291].x - lm[61].x)              # mouth corners
            face_w = abs(lm[454].x - lm[234].x) or 1
            if width / face_w > 0.38:
                smiles += 1
    cap.release()
    if not seen:
        return None
    return {"face_presence_pct": round(100 * faces / seen, 1), "smile_pct": round(100 * smiles / seen, 1)}
