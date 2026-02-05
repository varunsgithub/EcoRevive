import { useState, useCallback } from 'react'
import LandingPage from './components/LandingPage/LandingPage'
import InteractiveGlobe from './components/Globe/InteractiveGlobe'

/**
 * Main Application Component
 * Manages view transitions between Landing Page and Interactive Globe
 * Passes user type (personal/professional) to analysis components
 */
function App() {
  const [currentView, setCurrentView] = useState('landing') // 'landing' | 'globe'
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [userType, setUserType] = useState('personal') // 'personal' | 'professional'

  const handleEnterGlobe = useCallback((selectedUserType) => {
    setUserType(selectedUserType || 'personal')
    setIsTransitioning(true)
    // Smooth transition delay
    setTimeout(() => {
      setCurrentView('globe')
      setIsTransitioning(false)
    }, 800)
  }, [])

  const handleBackToLanding = useCallback(() => {
    // Switch immediately - no transition (fade-out would make landing invisible)
    setCurrentView('landing')
  }, [])

  return (
    <div className={`app ${isTransitioning ? 'transitioning' : ''}`}>
      {currentView === 'landing' ? (
        <LandingPage
          onEnterGlobe={handleEnterGlobe}
          isTransitioning={isTransitioning}
        />
      ) : (
        <>
          <InteractiveGlobe
            onBack={handleBackToLanding}
            isTransitioning={isTransitioning}
            userType={userType}
          />
          {/* Back button rendered at App level - outside Cesium entirely */}
          <button
            type="button"
            onClick={() => handleBackToLanding()}
            style={{
              position: 'fixed',
              top: '18px',
              left: '20px',
              zIndex: 2147483647,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 14px',
              background: 'rgba(20, 20, 20, 0.95)',
              border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: '8px',
              color: 'rgba(255,255,255,0.8)',
              fontSize: '13px',
              fontWeight: '500',
              cursor: 'pointer',
              pointerEvents: 'auto',
              backdropFilter: 'blur(8px)',
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
            Back
          </button>
        </>
      )}

      {/* Transition overlay */}
      <div className={`transition-overlay ${isTransitioning ? 'active' : ''}`}>
        <div className="transition-logo">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
            <path d="M12 8v8M8 12h8" />
          </svg>
          <span className="transition-text">EcoRevive</span>
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
          background: linear-gradient(135deg, #050508 0%, #0a1a12 50%, #050508 100%);
          opacity: 0;
          pointer-events: none;
          z-index: var(--z-overlay);
          transition: opacity 0.6s var(--ease-smooth);
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 16px;
        }

        .transition-logo {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 16px;
          color: var(--color-accent);
          opacity: 0;
          transform: scale(0.8);
          transition: all 0.4s var(--ease-bounce);
        }

        .transition-text {
          font-family: var(--font-display);
          font-size: 1.5rem;
          font-weight: 700;
          letter-spacing: -0.02em;
        }

        .transition-overlay.active {
          opacity: 1;
          pointer-events: all;
        }

        .transition-overlay.active .transition-logo {
           opacity: 1;
           transform: scale(1);
           animation: pulse-glow 1.5s ease-in-out infinite;
        }

        @keyframes pulse-glow {
          0%, 100% { filter: drop-shadow(0 0 10px rgba(0, 212, 170, 0.3)); }
          50% { filter: drop-shadow(0 0 30px rgba(0, 212, 170, 0.6)); }
        }
      `}</style>
    </div>
  )
}

export default App
