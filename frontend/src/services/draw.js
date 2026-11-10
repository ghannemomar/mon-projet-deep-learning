// Dessin des boîtes sur un canvas. Une couleur fixe par classe.
const COLORS = [
  '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
  '#bcf60c', '#fabebe', '#ff8c00', '#e6beff', '#00bfff', '#aaffc3', '#808000', '#ffd8b1', '#7fff00',
]

export const colorFor = (classId) => COLORS[classId % COLORS.length]

export function drawDetections(ctx, detections, violations = []) {
  const lw = Math.max(2, Math.round(ctx.canvas.width / 400))
  const font = Math.max(12, Math.round(ctx.canvas.width / 70))
  ctx.font = `${font}px sans-serif`
  ctx.textBaseline = 'top'
  for (const d of detections) {
    const { x1, y1, x2, y2 } = d.box
    const color = colorFor(d.class_id)
    ctx.strokeStyle = color
    ctx.lineWidth = lw
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    const text = `${d.label} ${(d.confidence * 100).toFixed(0)}%`
    const w = ctx.measureText(text).width + 6
    const ty = y1 - font - 4 < 0 ? y1 : y1 - font - 4
    ctx.fillStyle = color
    ctx.fillRect(x1, ty, w, font + 4)
    ctx.fillStyle = '#000'
    ctx.fillText(text, x1 + 3, ty + 2)
  }
  // têtes sans casque : contour rouge épais
  for (const v of violations) {
    ctx.strokeStyle = '#ff0000'
    ctx.lineWidth = lw * 2
    ctx.setLineDash([lw * 3, lw * 2])
    ctx.strokeRect(v.x1, v.y1, v.x2 - v.x1, v.y2 - v.y1)
    ctx.setLineDash([])
  }
}
