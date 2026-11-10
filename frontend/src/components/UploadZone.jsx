import { useState } from 'react'

export default function UploadZone({ onFile }) {
  const [over, setOver] = useState(false)
  const pick = (files) => {
    const f = files?.[0]
    if (f && f.type.startsWith('image/')) onFile(f)
  }
  return (
    <label
      className={`dropzone ${over ? 'over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files) }}
    >
      <input type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files)} />
      <span>Glisse une image ici ou <u>clique pour choisir</u></span>
    </label>
  )
}
