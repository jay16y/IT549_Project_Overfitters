import React, { useState, useRef } from 'react'
import { Upload, Camera, Image as ImageIcon } from 'lucide-react'

function UploadZone({ onUpload }) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)
  const cameraInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      onUpload(file)
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      onUpload(file)
    }
  }

  return (
    <div className="space-y-8">
      {/* Hero text */}
      <div className="text-center space-y-3">
        <h2 className="font-display text-3xl sm:text-4xl text-surface-900">
          Identify Your Pill
        </h2>
        <p className="text-surface-300 text-lg max-w-md mx-auto leading-relaxed">
          Take a photo or upload an image of your pill.
          Our AI will identify it in seconds.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`
          relative cursor-pointer rounded-3xl border-2 border-dashed
          transition-all duration-300 p-12 sm:p-16
          ${isDragging
            ? 'border-pill-500 bg-pill-50 upload-zone-active scale-[1.02]'
            : 'border-surface-200 bg-white hover:border-pill-400 hover:bg-pill-50/30'
          }
        `}
      >
        <div className="flex flex-col items-center gap-5">
          <div className={`
            w-20 h-20 rounded-2xl flex items-center justify-center
            transition-colors duration-300
            ${isDragging ? 'bg-pill-500 text-white' : 'bg-surface-100 text-surface-300'}
          `}>
            <Upload className="w-9 h-9" />
          </div>

          <div className="text-center space-y-2">
            <p className="text-surface-800 font-semibold text-lg">
              {isDragging ? 'Drop your image here' : 'Drag & drop a pill image'}
            </p>
            <p className="text-surface-300 text-sm">
              or click to browse • JPEG, PNG, WebP
            </p>
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Buttons */}
      <div className="flex flex-col sm:flex-row gap-3 justify-center">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center justify-center gap-2 px-6 py-3.5
                     bg-pill-600 text-white rounded-xl font-semibold
                     hover:bg-pill-700 transition-colors shadow-md shadow-pill-600/20"
        >
          <ImageIcon className="w-5 h-5" />
          Upload Image
        </button>

        <button
          onClick={() => cameraInputRef.current?.click()}
          className="flex items-center justify-center gap-2 px-6 py-3.5
                     bg-white text-surface-800 rounded-xl font-semibold
                     border border-surface-200 hover:border-pill-400
                     hover:text-pill-700 transition-colors"
        >
          <Camera className="w-5 h-5" />
          Take Photo
        </button>

        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Info cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4">
        {[
          { title: 'Fast', desc: 'Results in under 2 seconds', icon: '⚡' },
          { title: 'Accurate', desc: '73.9% top-1, 87% top-5', icon: '🎯' },
          { title: '2,047 Pills', desc: 'FDA-approved medications', icon: '💊' },
        ].map((item) => (
          <div key={item.title}
               className="bg-white rounded-xl p-4 border border-surface-100
                          text-center space-y-1">
            <span className="text-2xl">{item.icon}</span>
            <p className="font-semibold text-surface-800 text-sm">{item.title}</p>
            <p className="text-surface-300 text-xs">{item.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default UploadZone
