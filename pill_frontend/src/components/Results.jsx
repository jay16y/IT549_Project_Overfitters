import React from 'react'
import { RotateCcw, Clock, Circle, Square, Hexagon, Triangle, Pill } from 'lucide-react'

function getShapeIcon(shape) {
  const s = (shape || '').toLowerCase()
  if (s.includes('round') || s.includes('circle')) return <Circle className="w-4 h-4" />
  if (s.includes('square') || s.includes('rectangle')) return <Square className="w-4 h-4" />
  if (s.includes('hexagon')) return <Hexagon className="w-4 h-4" />
  if (s.includes('triangle')) return <Triangle className="w-4 h-4" />
  if (s.includes('capsule') || s.includes('oval')) return <Pill className="w-4 h-4" />
  return <Circle className="w-4 h-4" />
}

function getSimilarityColor(sim) {
  if (sim >= 80) return 'bg-green-500'
  if (sim >= 60) return 'bg-emerald-500'
  if (sim >= 40) return 'bg-yellow-500'
  return 'bg-orange-500'
}

function getSimilarityLabel(sim) {
  if (sim >= 80) return 'Excellent match'
  if (sim >= 60) return 'Good match'
  if (sim >= 40) return 'Possible match'
  return 'Low confidence'
}

function formatColors(colors) {
  if (!colors || colors === '[]' || colors === '') return 'N/A'
  try {
    const parsed = JSON.parse(colors.replace(/'/g, '"'))
    if (Array.isArray(parsed)) return parsed.join(', ')
    return String(colors)
  } catch {
    return String(colors).replace(/[\[\]']/g, '')
  }
}

function ResultCard({ result, index }) {
  const isTop = index === 0
  const hasRefImage = result.ref_image_url &&
    result.ref_image_url !== '' &&
    result.ref_image_url !== 'nan' &&
    result.ref_image_url !== 'undefined'

  return (
    <div
      className={`
        animate-fade-in-up opacity-0 stagger-${index + 1}
        rounded-2xl border transition-all duration-200
        ${isTop
          ? 'bg-white border-pill-200 shadow-lg shadow-pill-500/10 ring-1 ring-pill-100'
          : 'bg-white border-surface-100 hover:border-surface-200 hover:shadow-md'
        }
      `}
    >
      <div className="p-5 sm:p-6">
        {/* Top row: rank + name + similarity badge */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <span className={`
              w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold flex-shrink-0
              ${isTop
                ? 'bg-pill-600 text-white'
                : 'bg-surface-100 text-surface-800'
              }
            `}>
              {result.rank}
            </span>
            <div>
              <h3 className={`font-semibold leading-tight ${isTop ? 'text-lg' : 'text-base'} text-surface-900`}>
                {result.drug_name || 'Unknown'}
              </h3>
              {result.ndc && (
                <p className="text-xs text-surface-300 mt-0.5 font-mono">
                  NDC: {result.ndc}
                </p>
              )}
            </div>
          </div>

          {isTop && (
            <span className="px-2.5 py-1 bg-pill-50 text-pill-700 text-xs font-bold rounded-lg uppercase tracking-wide flex-shrink-0">
              Best Match
            </span>
          )}
        </div>

        {/* Reference image */}
        {hasRefImage && (
          <div className="mb-4 flex justify-center">
            <div className="relative">
              <img
                src={result.ref_image_url}
                alt={result.drug_name}
                className="w-36 h-36 object-contain rounded-xl border border-surface-100 bg-surface-50"
                onError={(e) => { e.target.parentElement.style.display = 'none' }}
              />
              <span className="absolute bottom-1 right-1 text-xs bg-black/50 text-white px-1.5 py-0.5 rounded-md">
                Reference
              </span>
            </div>
          </div>
        )}

        {/* Similarity bar */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-medium text-surface-300">
              {getSimilarityLabel(result.similarity)}
            </span>
            <span className="text-sm font-bold text-surface-800">
              {result.similarity}%
            </span>
          </div>
          <div className="h-2 bg-surface-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full similarity-bar-fill ${getSimilarityColor(result.similarity)}`}
              style={{ '--fill-width': `${result.similarity}%` }}
            />
          </div>
        </div>

        {/* Pill details grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {result.shape && result.shape !== '' && result.shape !== 'nan' && (
            <div className="bg-surface-50 rounded-lg p-2.5">
              <div className="flex items-center gap-1.5 text-surface-300 mb-1">
                {getShapeIcon(result.shape)}
                <span className="text-xs font-medium">Shape</span>
              </div>
              <p className="text-sm font-semibold text-surface-800 capitalize">
                {result.shape}
              </p>
            </div>
          )}

          {result.colors && result.colors !== '' && result.colors !== '[]' && result.colors !== 'nan' && (
            <div className="bg-surface-50 rounded-lg p-2.5">
              <div className="flex items-center gap-1.5 text-surface-300 mb-1">
                <div className="w-4 h-4 rounded-full bg-gradient-to-br from-red-400 via-blue-400 to-green-400" />
                <span className="text-xs font-medium">Color</span>
              </div>
              <p className="text-sm font-semibold text-surface-800 capitalize">
                {formatColors(result.colors)}
              </p>
            </div>
          )}

          {result.imprint && result.imprint !== '' && result.imprint !== 'nan' && (
            <div className="bg-surface-50 rounded-lg p-2.5">
              <div className="flex items-center gap-1.5 text-surface-300 mb-1">
                <span className="text-xs">🔤</span>
                <span className="text-xs font-medium">Imprint</span>
              </div>
              <p className="text-sm font-semibold text-surface-800 font-mono">
                {result.imprint}
              </p>
            </div>
          )}

          {result.size_mm && result.size_mm !== '' && result.size_mm !== 'nan' && (
            <div className="bg-surface-50 rounded-lg p-2.5">
              <div className="flex items-center gap-1.5 text-surface-300 mb-1">
                <span className="text-xs">📏</span>
                <span className="text-xs font-medium">Size</span>
              </div>
              <p className="text-sm font-semibold text-surface-800">
                {result.size_mm} mm
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


function Results({ results, preview, inferenceTime, onReset }) {
  return (
    <div className="space-y-6">
      {/* Header with image + stats */}
      <div className="flex flex-col sm:flex-row items-center gap-6 pb-2">
        {preview && (
          <img
            src={preview}
            alt="Uploaded pill"
            className="w-28 h-28 object-cover rounded-2xl shadow-lg border-2 border-white"
          />
        )}
        <div className="text-center sm:text-left space-y-2">
          <h2 className="font-display text-2xl sm:text-3xl text-surface-900">
            Results Found
          </h2>
          <div className="flex flex-wrap items-center justify-center sm:justify-start gap-3">
            <span className="flex items-center gap-1.5 text-sm text-surface-300">
              <Clock className="w-4 h-4" />
              {inferenceTime.toFixed(0)} ms
            </span>
            <span className="text-surface-200">•</span>
            <span className="text-sm text-surface-300">
              {results.length} matches
            </span>
          </div>
        </div>
        <div className="sm:ml-auto">
          <button
            onClick={onReset}
            className="flex items-center gap-2 px-5 py-2.5
                       bg-surface-800 text-white rounded-xl font-medium
                       hover:bg-surface-900 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            New Scan
          </button>
        </div>
      </div>

      {/* Result cards */}
      <div className="space-y-3">
        {results.map((result, index) => (
          <ResultCard key={result.pill_id} result={result} index={index} />
        ))}
      </div>

      {/* Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mt-6">
        <p className="text-amber-800 text-sm leading-relaxed">
          <span className="font-semibold">⚠️ Disclaimer:</span> This tool is for
          informational purposes only. Always consult a healthcare professional
          or pharmacist for accurate pill identification.
        </p>
      </div>
    </div>
  )
}

export default Results
