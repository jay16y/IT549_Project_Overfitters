import React from 'react'
import { Search } from 'lucide-react'

function LoadingState({ preview }) {
  return (
    <div className="flex flex-col items-center gap-8 animate-fade-in-up">
      {/* Image with scan effect */}
      <div className="relative">
        {preview && (
          <div className="relative w-56 h-56 rounded-2xl overflow-hidden shadow-xl">
            <img
              src={preview}
              alt="Uploaded pill"
              className="w-full h-full object-cover"
            />
            {/* Scanning overlay */}
            <div className="absolute inset-0 bg-gradient-to-b from-pill-500/10 to-pill-500/5" />
            <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-pill-500 to-transparent scan-line opacity-80" />
            {/* Corner brackets */}
            <div className="absolute top-3 left-3 w-6 h-6 border-t-2 border-l-2 border-pill-500 rounded-tl" />
            <div className="absolute top-3 right-3 w-6 h-6 border-t-2 border-r-2 border-pill-500 rounded-tr" />
            <div className="absolute bottom-3 left-3 w-6 h-6 border-b-2 border-l-2 border-pill-500 rounded-bl" />
            <div className="absolute bottom-3 right-3 w-6 h-6 border-b-2 border-r-2 border-pill-500 rounded-br" />
          </div>
        )}
      </div>

      {/* Loading text */}
      <div className="text-center space-y-3">
        <div className="flex items-center justify-center gap-3">
          <Search className="w-5 h-5 text-pill-600 animate-pulse" />
          <p className="text-surface-800 font-semibold text-lg">Analyzing pill...</p>
        </div>
        <p className="text-surface-300 text-sm">
          Searching across 2,047 medications
        </p>

        {/* Bouncing dots */}
        <div className="flex justify-center gap-1.5 pt-2">
          <div className="w-2.5 h-2.5 rounded-full bg-pill-500 dot-1" />
          <div className="w-2.5 h-2.5 rounded-full bg-pill-500 dot-2" />
          <div className="w-2.5 h-2.5 rounded-full bg-pill-500 dot-3" />
        </div>
      </div>
    </div>
  )
}

export default LoadingState
