import React from 'react'

function Footer() {
  return (
    <footer className="border-t border-surface-100 mt-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-sm text-surface-300">
          <p>
            Built with <span className="text-pill-600 font-semibold">DINOv2</span> +
            <span className="text-pill-600 font-semibold"> LoRA</span> +
            <span className="text-pill-600 font-semibold"> Sub-center ArcFace</span>
          </p>
          <p>
            Dataset: NLM C3PI • 2,047 pill types • {new Date().getFullYear()}
          </p>
        </div>
      </div>
    </footer>
  )
}

export default Footer
