import { useEffect, useRef } from 'react'
import { drawDetections } from '../services/draw'

// Affiche l'image envoyée et dessine les boîtes renvoyées par l'API par-dessus
export default function ImageView({ src, result }) {
  const canvasRef = useRef(null)
  useEffect(() => {
    if (!src) return
    const img = new Image()
    img.onload = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0)
      if (result) drawDetections(ctx, result.detections, result.compliance.violations)
    }
    img.src = src
  }, [src, result])
  if (!src) return <div className="placeholder">Aucune image</div>
  return <canvas ref={canvasRef} className="view" />
}
