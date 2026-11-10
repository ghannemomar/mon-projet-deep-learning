export default function AlertBanner({ compliance }) {
  if (!compliance) return null
  const cls = compliance.alert ? 'banner alert' : compliance.heads ? 'banner ok' : 'banner neutral'
  return (
    <div className={cls} role="status">
      <strong>{compliance.alert ? '⚠ NON CONFORME' : compliance.heads ? '✔ CONFORME' : 'ℹ'}</strong>
      <span>{compliance.message}</span>
    </div>
  )
}
