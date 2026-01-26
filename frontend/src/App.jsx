import { useState, useCallback } from 'react'
import LandingPage from './components/LandingPage/LandingPage'
import InteractiveGlobe from './components/Globe/InteractiveGlobe'

/**
 * Main Application Component
 * Manages view transitions between Landing Page and Interactive Globe
 */
function App() {
  const [currentView, setCurrentView] = useState('landing') // 'landing' | 'globe'
  const [isTransitioning, setIsTransitioning] = useState(false)

  const handleEnterGlobe = useCallback(() => {
    setIsTransitioning(true)
    // Smooth transition delay
    setTimeout(() => {
      setCurrentView('globe')
      setIsTransitioning(false)
    }, 800)
  }, [])

  const handleBackToLanding = useCallback(() => {
    setIsTransitioning(true)
    setTimeout(() => {
      setCurrentView('landing')
      setIsTransitioning(false)
    }, 600)
  }, [])

  return (
    <div className={`app ${isTransitioning ? 'transitioning' : ''}`}>
      {currentView === 'landing' ? (
        <LandingPage
          onEnterGlobe={handleEnterGlobe}
          isTransitioning={isTransitioning}
        />
      ) : (
        <InteractiveGlobe
          onBack={handleBackToLanding}
          isTransitioning={isTransitioning}
        />
      )}

      {/* Transition overlay */}
      <div className={`transition-overlay ${isTransitioning ? 'active' : ''}`}>
        <div className="transition-logo">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
            <path d="M12 8v8M8 12h8" />
          </svg>
        </div>
      </div>

      <style>{`
        .app {
          min-height: 100vh;
          position: relative;
        }
        
        .transition-overlay {
          position: fixed;
          inset: 0;
          background: var(--color-bg-primary);
          opacity: 0;
          pointer-events: none;
          z-index: var(--z-overlay);
          transition: opacity 0.6s var(--ease-smooth);
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .transition-logo {
          color: var(--color-accent);
          opacity: 0;
          transform: scale(0.8);
          transition: all 0.4s var(--ease-bounce);
        }

        .transition-overlay.active {
          opacity: 1;
          pointer-events: all;
        }

        .transition-overlay.active .transition-logo {
           opacity: 1;
           transform: scale(1);
        }
      `}</style>
    </div>
  )
}

export default App
