import { ChangeEvent, DragEvent, useMemo, useState } from "react";

type Prediction = {
  prediction: "NORMAL" | "PNEUMONIA";
  confidence: number;
  probabilities: { NORMAL: number; PNEUMONIA: number };
  request_id: string;
};

const API_URL = import.meta.env.VITE_API_URL ?? "https://chest-disease-api.onrender.com";
const MAX_BYTES = 10 * 1024 * 1024;

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);

  function selectFile(candidate: File | null) {
    setError("");
    setResult(null);
    if (!candidate) return;
    if (!["image/jpeg", "image/png"].includes(candidate.type)) {
      setError("Please choose a JPG or PNG image.");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setError("Image must be smaller than 10 MB.");
      return;
    }
    setFile(candidate);
    setPreview(URL.createObjectURL(candidate));
  }

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    selectFile(e.target.files?.[0] ?? null);
    e.target.value = "";
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDrag(false);
    selectFile(e.dataTransfer.files?.[0] ?? null);
  }

  function clear() {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
  }

  async function analyze() {
    if (!file) {
      setError("Upload a chest X-ray first.");
      return;
    }

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const body = new FormData();
      body.append("image", file);
      const response = await fetch(API_URL + "/predict", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error ?? "Prediction failed.");
      setResult(data as Prediction);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to reach the prediction API.");
    } finally {
      setBusy(false);
    }
  }

  const confidence = useMemo(
    () => (result ? (result.confidence * 100).toFixed(1) + "%" : "—"),
    [result],
  );

  return (
    <div className="page">
      <header className="nav">
        <a className="logo" href="/">CHEST<span>/</span>DISEASE AI</a>
        <div className="nav-right">
          <span className="live"><i /> API LIVE</span>
          <a href="https://github.com/Nidhi1310/Chest-Disease-" target="_blank" rel="noreferrer">GitHub ↗</a>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="eyebrow">VGG16 TRANSFER-LEARNING DEMO</div>
          <h1>Chest X-ray<br /><em>classification,</em> made clear.</h1>
          <p>Upload a chest X-ray to interact with the deployed NORMAL vs PNEUMONIA classification model.</p>
          <div className="tags"><b>224 × 224</b><b>JPG / PNG</b><b>REST API</b><b>RENDER</b></div>
        </section>

        <section className="grid">
          <article className="card">
            <div className="label">01 · INPUT</div>
            <div className="card-head">
              <h2>Upload an X-ray</h2>
              {file && <button onClick={clear}>Clear</button>}
            </div>

            {!file ? (
              <div
                className={"drop " + (drag ? "active" : "")}
                onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={onDrop}
              >
                <input id="xray" type="file" accept="image/jpeg,image/png" onChange={onChange} />
                <label htmlFor="xray">
                  <div className="upload-symbol">＋</div>
                  <strong>Drop an X-ray here</strong>
                  <span>or click to browse</span>
                  <small>JPG or PNG · up to 10 MB</small>
                </label>
              </div>
            ) : (
              <div className="preview">
                <img src={preview} alt="Selected chest X-ray" />
                <div className="filename">
                  {file.name}
                  <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
              </div>
            )}

            {error && <div className="error">{error}</div>}

            <button className="primary" onClick={analyze} disabled={!file || busy}>
              {busy ? "Analyzing…" : "Run analysis →"}
            </button>
            <p className="tiny">Images are sent to the prediction API for this request only.</p>
          </article>

          <article className="card result">
            <div className="label">02 · OUTPUT</div>
            <div className="card-head"><h2>Model result</h2><span className="badge">VGG16</span></div>

            {!result ? (
              <div className="waiting">
                <div className="pulse">◎</div>
                <h3>Waiting for an image</h3>
                <p>Your classification result will appear here after analysis.</p>
              </div>
            ) : (
              <div className="result-body">
                <div className={"answer " + result.prediction.toLowerCase()}>
                  <span>CLASSIFICATION</span>
                  <strong>{result.prediction}</strong>
                </div>
                <div className="confidence">
                  <div><span>Confidence</span><strong>{confidence}</strong></div>
                  <div className="bar"><i style={{ width: (result.confidence * 100) + "%" }} /></div>
                </div>
                <div className="prob-grid">
                  <div><span>NORMAL</span><strong>{(result.probabilities.NORMAL * 100).toFixed(1)}%</strong></div>
                  <div><span>PNEUMONIA</span><strong>{(result.probabilities.PNEUMONIA * 100).toFixed(1)}%</strong></div>
                </div>
                <div className="request">Request <code>{result.request_id}</code></div>
              </div>
            )}

            <div className="disclaimer">
              <strong>Educational use only.</strong>
              <span>This model is not a clinical diagnostic system and should not be used for medical decisions.</span>
            </div>
          </article>
        </section>

        <section className="footer-strip">
          <div>
            <span>PIPELINE</span>
            <strong>Patient-level split → VGG16 → Medical evaluation → Flask API</strong>
          </div>
          <a href={API_URL + "/docs"} target="_blank" rel="noreferrer">API docs ↗</a>
        </section>
      </main>

      <footer><span>Chest Disease AI · Educational portfolio project</span><span>© 2026</span></footer>
    </div>
  );
}
