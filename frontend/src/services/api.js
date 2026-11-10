// Appels à l'API FastAPI. L'URL vient de la variable VITE_API_URL (fixée au build Docker).
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail
    } catch {
      /* corps non JSON */
    }
    throw new Error(`${res.status} — ${detail}`)
  }
  return res
}

export async function getHealth() {
  return (await handle(await fetch(`${API_URL}/health`))).json()
}

export async function getMetadata() {
  return (await handle(await fetch(`${API_URL}/model/metadata`))).json()
}

export async function predict(blob, conf = 0.25) {
  const form = new FormData()
  form.append('file', blob, 'image.jpg')
  const res = await fetch(`${API_URL}/predict?conf=${conf}`, { method: 'POST', body: form })
  return (await handle(res)).json()
}

export async function predictVisualize(blob, conf = 0.25) {
  const form = new FormData()
  form.append('file', blob, 'image.jpg')
  const res = await fetch(`${API_URL}/predict/visualize?conf=${conf}`, { method: 'POST', body: form })
  return URL.createObjectURL(await (await handle(res)).blob())
}
