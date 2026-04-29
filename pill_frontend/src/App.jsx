import React, { useState } from 'react'
import Header from './components/Header'
import UploadZone from './components/UploadZone'
import LoadingState from './components/LoadingState'
import Results from './components/Results'
import Footer from './components/Footer'

const API_URL = '/api'

function App() {
  const [state, setState] = useState('idle')  // idle | loading | results | error
  const [preview, setPreview] = useState(null)
  const [results, setResults] = useState([])
  const [inferenceTime, setInferenceTime] = useState(0)
  const [error, setError] = useState('')

  const handleUpload = async (file) => {
    // Show preview
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)

    // Start loading
    setState('loading')
    setError('')

    // Send to API
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Prediction failed')
      }

      const data = await response.json()
      setResults(data.results)
      setInferenceTime(data.inference_time_ms)
      setState('results')
    } catch (err) {
      setError(err.message || 'Failed to connect to server')
      setState('error')
    }
  }

  const handleReset = () => {
    setState('idle')
    setPreview(null)
    setResults([])
    setError('')
  }

  return (
    <div className="min-h-screen bg-gradient-mesh">
      <Header />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        {state === 'idle' && (
          <UploadZone onUpload={handleUpload} />
        )}

        {state === 'loading' && (
          <LoadingState preview={preview} />
        )}

        {state === 'results' && (
          <Results
            results={results}
            preview={preview}
            inferenceTime={inferenceTime}
            onReset={handleReset}
          />
        )}

        {state === 'error' && (
          <div className="text-center space-y-6">
            {preview && (
              <img
                src={preview}
                alt="Uploaded pill"
                className="w-48 h-48 object-cover rounded-2xl mx-auto shadow-lg"
              />
            )}
            <div className="bg-red-50 border border-red-200 rounded-2xl p-6 max-w-md mx-auto">
              <p className="text-red-700 font-medium text-lg">Something went wrong</p>
              <p className="text-red-500 mt-2">{error}</p>
            </div>
            <button
              onClick={handleReset}
              className="px-6 py-3 bg-surface-800 text-white rounded-xl font-medium
                         hover:bg-surface-900 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}

export default App
