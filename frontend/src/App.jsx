import { useEffect, useState } from 'react'
import { API_URL, getHealth, getMetadata, predict, predictVisualize } from './services/api'
import UploadZone from './components/UploadZone'
import ExampleGallery from './components/ExampleGallery'
import ImageView from './components/ImageView'
import WebcamView from './components/WebcamView'
import AlertBanner from './components/AlertBanner'
import LatencyPanel from './components/LatencyPanel'
import DetectionList from './components/DetectionList'

export default function App() {
  const [mode, setMode] = useState('image')
  const [conf, setConf] = useState(0.25)
  const [health, setHealth] = useState(null)
  const [meta, setMeta] = useState(null)
  const [file, setFile] = useState(null)
  const [src, setSrc] = useState(null)
  const [serverImg, setServerImg] = useState(null)
  const [result, setResult] = useState(null)
  const [clientMs, setClientMs] = useState(null)
  const [fps, setFps] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getHealth()
      .then((h) => {
        setHealth(h)
        if (h.model_loaded) getMetadata().then(setMeta).catch(() => {})
      })
      .catch(() => setHealth({ status: 'injoignable', model_loaded: false }))
  }, [])

  async function analyse(f, c = conf) {
    if (!f) return
    setError(null)
    setLoading(true)
    setServerImg(null)
    try {
      const t0 = performance.now()
      const res = await predict(f, c)
      setClientMs(performance.now() - t0)
      setResult(res)
      setFps(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function onFile(f) {
    setFile(f)
    setSrc(URL.createObjectURL(f))
    setResult(null)
    analyse(f)
  }

  async function showServerImage() {
    try {
      setServerImg(await predictVisualize(file, conf))
    } catch (e) {
      setError(e.message)
    }
  }

  const switchMode = (m) => {
    setMode(m)
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <header>
        <h1>Détection d'EPI en temps réel</h1>
        <div className="status">
          <span className={`pill ${health?.model_loaded ? 'ok' : 'ko'}`}>
            API {health ? (health.model_loaded ? 'prête' : health.status === 'ok' ? 'sans modèle' : 'injoignable') : '…'}
          </span>
          {meta && (
            <span className="pill">
              {meta.name} v{meta.version} · {meta.source} · {meta.input_size[0]} px
            </span>
          )}
          <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">Swagger</a>
        </div>
      </header>

      <nav className="tabs">
        <button className={mode === 'image' ? 'active' : ''} onClick={() => switchMode('image')}>Image</button>
        <button className={mode === 'webcam' ? 'active' : ''} onClick={() => switchMode('webcam')}>Webcam</button>
        <label className="conf">
          Seuil de confiance : <b>{conf.toFixed(2)}</b>
          <input
            type="range" min="0.05" max="0.9" step="0.05" value={conf}
            onChange={(e) => setConf(Number(e.target.value))}
            onMouseUp={() => mode === 'image' && analyse(file)}
            onTouchEnd={() => mode === 'image' && analyse(file)}
          />
        </label>
      </nav>

      {error && <div className="banner alert">Erreur : {error}</div>}
      <AlertBanner compliance={result?.compliance} />

      <main>
        <section className="viewer">
          {mode === 'image' ? (
            <>
              <UploadZone onFile={onFile} />
              <ExampleGallery onFile={onFile} />
              {loading && <p className="muted">Analyse en cours…</p>}
              <ImageView src={src} result={result} />
              {file && result && (
                <button onClick={showServerImage}>Voir l'image annotée par le serveur (/predict/visualize)</button>
              )}
              {serverImg && <img className="view" src={serverImg} alt="annotée par le serveur" />}
            </>
          ) : (
            <WebcamView
              conf={conf}
              onResult={(res, ms, f) => { setResult(res); setClientMs(ms); setFps(f) }}
              onError={setError}
            />
          )}
        </section>
        <aside>
          <DetectionList detections={result?.detections} />
          <LatencyPanel result={result} clientMs={clientMs} fps={fps} />
        </aside>
      </main>
    </div>
  )
}
