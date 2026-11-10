import { useEffect, useState } from 'react'

// Images d'exemple : fichiers placés dans public/examples/ et listés dans index.json
export default function ExampleGallery({ onFile }) {
  const [items, setItems] = useState([])
  useEffect(() => {
    fetch('/examples/index.json')
      .then((r) => (r.ok ? r.json() : []))
      .then(setItems)
      .catch(() => setItems([]))
  }, [])
  if (!items.length) return null
  const choose = async (name) => {
    const blob = await (await fetch(`/examples/${name}`)).blob()
    onFile(new File([blob], name, { type: blob.type }))
  }
  return (
    <div className="gallery">
      {items.map((name) => (
        <button key={name} onClick={() => choose(name)} title={name}>
          <img src={`/examples/${name}`} alt={name} />
        </button>
      ))}
    </div>
  )
}
