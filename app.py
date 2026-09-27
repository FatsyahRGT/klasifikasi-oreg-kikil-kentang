import io
import os
import glob
import pickle
import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Klasifikasi Makanan Lokal - Kikil, Oreg, Kentang")

CLASSES_INFO = {
    "oreg": {"display": "Orek Tempe", "desc": "Olahan tempe gurih & manis"},
    "kentang": {"display": "Olahan Kentang / Balado", "desc": "Potongan kentang balado berbumbu"},
    "kikil": {"display": "Kikil Sapi", "desc": "Kikil sapi kenyal khas warteg"}
}

model = None
model_filename = None
model_meta = {}

def find_model_path():
    env_path = os.getenv("MODEL_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists("model_knn_final.pkl"):
        return "model_knn_final.pkl"
    if os.path.exists("model.pkl"):
        return "model.pkl"
    pkl_files = glob.glob("*.pkl")
    return pkl_files[0] if pkl_files else None

def inspect_model(m):
    meta = {
        "type": str(type(m)),
        "n_features_in": getattr(m, "n_features_in_", None),
        "classes": [str(c) for c in getattr(m, "classes_", [])],
        "scale_255": False
    }
    
    # Periksa data fit_X KNN jika tersedia
    fit_x = getattr(m, "_fit_X", None)
    if fit_x is not None:
        try:
            max_v = float(np.max(fit_x))
            min_v = float(np.min(fit_x))
            meta["fit_X_shape"] = list(fit_x.shape)
            meta["fit_X_max"] = max_v
            meta["fit_X_min"] = min_v
            # Jika data training bernilai > 1.5, berarti dilatih pada rentang [0, 255]
            if max_v > 1.5:
                meta["scale_255"] = True
        except Exception:
            pass
            
    return meta

def load_ml_model():
    global model, model_filename, model_meta
    path = find_model_path()
    if path:
        model_filename = path
        try:
            print(f"[*] Memuat model dari: {path}")
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
            except Exception:
                model = joblib.load(path)
            model_meta = inspect_model(model)
            print(f"[+] Model '{path}' berhasil dimuat! Metadata: {model_meta}")
        except Exception as e:
            print(f"[!] Gagal memuat model '{path}': {str(e)}")
            model = None
    else:
        print("[!] Tidak ada file .pkl yang ditemukan di direktori kerja.")
        model = None

@app.on_event("startup")
def startup_event():
    load_ml_model()

def preprocess_image_for_model(image_bytes: bytes, target_model) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        expected_features = getattr(target_model, "n_features_in_", None)
        scale_255 = model_meta.get("scale_255", False)
        
        if expected_features:
            # 1. 224x224x3 RGB (150528)
            if expected_features == 224 * 224 * 3:
                img = img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                arr = np.array(img, dtype=np.float32)
                if not scale_255:
                    arr = arr / 255.0
                return arr.reshape(1, -1)
                
            # 2. 224x224 Grayscale (50176)
            elif expected_features == 224 * 224:
                img = img.convert("L").resize((224, 224), Image.Resampling.LANCZOS)
                arr = np.array(img, dtype=np.float32)
                if not scale_255:
                    arr = arr / 255.0
                return arr.reshape(1, -1)
                
            # 3. Estimasi resolusi lain (e.g. 100x100x3, 64x64x3, 32x32x3)
            size_rgb = int(round(np.sqrt(expected_features / 3)))
            if size_rgb * size_rgb * 3 == expected_features:
                img = img.convert("RGB").resize((size_rgb, size_rgb), Image.Resampling.LANCZOS)
                arr = np.array(img, dtype=np.float32)
                if not scale_255:
                    arr = arr / 255.0
                return arr.reshape(1, -1)
            
            size_gray = int(round(np.sqrt(expected_features)))
            if size_gray * size_gray == expected_features:
                img = img.convert("L").resize((size_gray, size_gray), Image.Resampling.LANCZOS)
                arr = np.array(img, dtype=np.float32)
                if not scale_255:
                    arr = arr / 255.0
                return arr.reshape(1, -1)
            
            # Fallback
            img = img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32)
            if not scale_255:
                arr = arr / 255.0
            flat = arr.flatten()
            if len(flat) >= expected_features:
                return flat[:expected_features].reshape(1, -1)
            else:
                return np.pad(flat, (0, expected_features - len(flat))).reshape(1, -1)

        # Default standard 224x224 RGB
        img = img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        if not scale_255:
            arr = arr / 255.0
        return arr.reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses gambar: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    template_path = os.path.join("templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="File templates/index.html tidak ditemukan.")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "model_filename": model_filename,
        "metadata": model_meta
    }

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    global model
    if model is None:
        load_ml_model()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="File kosong.")

    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "File model (.pkl) belum ditemukan di server."
            }
        )

    features = preprocess_image_for_model(contents, model)

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
        elif hasattr(model, "predict"):
            preds = model.predict(features)
            pred_val = preds[0]
            classes_list = getattr(model, "classes_", ["oreg", "kentang", "kikil"])
            probs = [1.0 if str(c) == str(pred_val) else 0.0 for c in classes_list]
        elif callable(model):
            res = model(features)
            probs = res[0] if isinstance(res, (list, np.ndarray)) else [1.0, 0.0, 0.0]
        else:
            raise ValueError("Model tidak memiliki method predict.")

        probs = np.array(probs, dtype=float).flatten()

        # Normalisasi jika belum softmax/probabilitas
        if np.sum(probs) > 1.05 or np.min(probs) < 0:
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / np.sum(exp_p)
        elif np.sum(probs) > 0:
            probs = probs / np.sum(probs)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan saat inferensi model: {str(e)}")

    # Mapping kelas dinamis berdasarkan classes_ asli di model
    raw_classes = [str(c).lower() for c in getattr(model, "classes_", [])]
    results = []

    # Jika model memiliki classes_ spesifik
    if raw_classes:
        for idx, cls_name in enumerate(raw_classes):
            # Cocokkan nama kelas dengan informasi display
            matched_key = None
            for key in ["oreg", "kentang", "kikil"]:
                if key in cls_name:
                    matched_key = key
                    break
            
            display_name = CLASSES_INFO[matched_key]["display"] if matched_key else cls_name.capitalize()
            desc = CLASSES_INFO[matched_key]["desc"] if matched_key else "-"
            
            prob_val = float(probs[idx]) if idx < len(probs) else 0.0
            results.append({
                "id": matched_key or cls_name,
                "name": display_name,
                "desc": desc,
                "probability": round(prob_val * 100, 2),
                "raw_score": prob_val
            })
    else:
        target_keys = ["oreg", "kentang", "kikil"]
        for idx, key in enumerate(target_keys):
            prob_val = float(probs[idx]) if idx < len(probs) else 0.0
            info = CLASSES_INFO[key]
            results.append({
                "id": key,
                "name": info["display"],
                "desc": info["desc"],
                "probability": round(prob_val * 100, 2),
                "raw_score": prob_val
            })

    results.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "success": True,
        "model_used": model_filename,
        "top_prediction": results[0],
        "all_predictions": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)