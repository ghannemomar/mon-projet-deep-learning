import { useEffect, useRef, useState } from 'react'
import { predict } from '../services/api'
import { drawDetections } from '../services/draw'

// Webcam : on envoie une image à l'API, on attend la réponse, puis on envoie la suivante.
export default function WebcamView({ conf, onResult, onError }) {
  const videoRef = useRef(null)
  const overlayRef = useRef(null)
  const runningRef = useRef(false)
  const confRef = useRef(conf)
  const [running, setRunning] = useState(false)

  useEffect(() => { confRef.current = conf }, [conf])
  useEffect(() => () => stop(), [])  // arrêt propre si on quitte l'onglet

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      runningRef.current = true
      setRunning(true)
      loop()
    } catch (e) {
      onError(`Webcam inaccessible : ${e.message}`)
    }
  }

  function stop() {
    runningRef.current = false
    setRunning(false)
    const stream = videoRef.current?.srcObject
    stream?.getTracks().forEach((t) => t.stop())
    if (videoRef.current) videoRef.current.srcObject = null
  }

  async function loop() {
    const video = videoRef.current
    const capture = document.createElement('canvas')
    let last = performance.now()
    while (runningRef.current && video) {
      capture.width = video.videoWidth
      capture.height = video.videoHeight
      capture.getContext('2d').drawImage(video, 0, 0)
      const blob = await new Promise((r) => capture.toBlob(r, 'image/jpeg', 0.8))
      const t0 = performance.now()
      try {
        const res = await predict(blob, confRef.current)
        const now = performance.now()
        const overlay = overlayRef.current
        if (!overlay || !runningRef.current) break
        overlay.width = capture.width
        overlay.height = capture.height
        const ctx = overlay.getContext('2d')
        ctx.clearRect(0, 0, overlay.width, overlay.height)
        drawDetections(ctx, res.detections, res.compliance.violations)
        onResult(res, now - t0, 1000 / (now - last))
        last = now
      } catch (e) {
        onError(e.message)
        stop()
      }
    }
  }

  return (
    <div>
      <div className="webcam">
        <video ref={videoRef} muted playsInline />
        <canvas ref={overlayRef} />
        {!running && <div className="placeholder overlay-msg">Webcam arrêtée</div>}
      </div>
      <button className="primary" onClick={running ? stop : start}>
        {running ? '■ Arrêter la webcam' : '▶ Démarrer la webcam'}
      </button>
    </div>
  )
}
