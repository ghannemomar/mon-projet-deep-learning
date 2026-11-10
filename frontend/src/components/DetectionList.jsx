import { colorFor } from '../services/draw'

export default function DetectionList({ detections }) {
  if (!detections) return null
  const counts = {}
  for (const d of detections) counts[d.label] = { n: (counts[d.label]?.n || 0) + 1, id: d.class_id }
  const entries = Object.entries(counts).sort((a, b) => b[1].n - a[1].n)
  return (
    <div className="panel">
      <h3>Objets détectés ({detections.length})</h3>
      {entries.length === 0 && <p className="muted">Aucun objet au-dessus du seuil.</p>}
      <ul className="chips">
        {entries.map(([label, { n, id }]) => (
          <li key={label} style={{ borderColor: colorFor(id) }}>
            <span className="dot" style={{ background: colorFor(id) }} /> {label} × {n}
          </li>
        ))}
      </ul>
    </div>
  )
}
