export default function LatencyPanel({ result, clientMs, fps }) {
  if (!result) return null
  const l = result.latency_ms
  return (
    <div className="panel">
      <h3>Latence</h3>
      <table>
        <tbody>
          <tr><td>Client (aller-retour réseau compris)</td><td>{clientMs?.toFixed(0)} ms</td></tr>
          <tr><td>Serveur total</td><td>{l.total.toFixed(1)} ms</td></tr>
          <tr><td>— prétraitement</td><td>{l.preprocess.toFixed(1)} ms</td></tr>
          <tr><td>— inférence ONNX</td><td>{l.inference.toFixed(1)} ms</td></tr>
          <tr><td>— post-traitement (NMS)</td><td>{l.postprocess.toFixed(1)} ms</td></tr>
          {fps ? <tr><td>Images par seconde (webcam)</td><td>{fps.toFixed(1)}</td></tr> : null}
        </tbody>
      </table>
    </div>
  )
}
