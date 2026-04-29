import React from 'react'
import { Pill, Shield } from 'lucide-react'

function Header() {
  return (
    <header className="border-b border-surface-200 bg-white/60 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-pill-600 flex items-center justify-center shadow-md">
            <Pill className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-display text-xl text-surface-900">PillScan</h1>
            <p className="text-xs text-surface-300 font-medium tracking-wide uppercase">
              AI Pill Recognition
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-pill-700 bg-pill-50 px-3 py-1.5 rounded-lg">
          <Shield className="w-4 h-4" />
          <span className="text-xs font-semibold">2,047 Pills</span>
        </div>
      </div>
    </header>
  )
}

export default Header
