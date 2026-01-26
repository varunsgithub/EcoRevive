import { useState, useEffect, useRef } from 'react'
import HeroEarth from './HeroEarth'
import './LandingPage.css'

/**
 * Landing Page Component
 * Features full-screen 3D Earth hero with scroll-reveal About section
 */
export default function LandingPage({ onEnterGlobe, isTransitioning }) {
    const [scrollY, setScrollY] = useState(0)
    const [isAboutVisible, setIsAboutVisible] = useState(false)
    const [showModal, setShowModal] = useState(false)
    const [selectedUserType, setSelectedUserType] = useState(null)
    const aboutRef = useRef(null)

    // Handle click on explore button - show modal instead of entering directly
    const handleExploreClick = () => {
        setShowModal(true)
    }

    // Handle user type selection
    const handleUserTypeSelect = (userType) => {
        setSelectedUserType(userType)
        console.log('User selected:', userType)
        // Close modal and enter globe after a brief delay
        setTimeout(() => {
            setShowModal(false)
            onEnterGlobe(userType)
        }, 300)
    }

    // Close modal
    const handleCloseModal = () => {
        setShowModal(false)
    }

    useEffect(() => {
        const handleScroll = () => {
            setScrollY(window.scrollY)
        }

        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        setIsAboutVisible(true)
                    }
                })
            },
            { threshold: 0.2 }
        )

        if (aboutRef.current) {
            observer.observe(aboutRef.current)
        }

        window.addEventListener('scroll', handleScroll, { passive: true })

        return () => {
            window.removeEventListener('scroll', handleScroll)
            observer.disconnect()
        }
    }, [])

    // Calculate parallax effect
    const heroTransform = `translateY(${scrollY * 0.3}px)`
    const heroOpacity = Math.max(0, 1 - scrollY / 600)

    return (
        <div className={`landing-page ${isTransitioning ? 'transitioning' : ''}`}>
            {/* Hero Section with 3D Earth */}
            <section className="hero-section">
                <div
                    className="hero-content"
                    style={{
                        transform: heroTransform,
                        opacity: heroOpacity
                    }}
                >
                    {/* Branding */}
                    <div className="hero-header">
                        <div className="logo">
                            <span className="logo-icon">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
                                    <path d="M12 8v8M8 12h8" />
                                </svg>
                            </span>
                            <span className="logo-text">EcoRevive</span>
                        </div>
                    </div>

                    {/* Main headline */}
                    <div className="hero-text">
                        <h1 className="hero-title">
                            <span className="title-line">Restore</span>
                            <span className="title-line text-gradient">Our Planet</span>
                        </h1>
                        <p className="hero-subtitle">
                            AI-powered ecosystem restoration from satellite imagery
                        </p>
                    </div>

                    {/* 3D Earth */}
                    <div className="earth-container">
                        <HeroEarth onClick={handleExploreClick} />
                    </div>
                </div>

                {/* Scroll indicator */}
                <div className="scroll-indicator">
                    <div className="scroll-line" />
                    <span className="scroll-text">Scroll to learn more</span>
                </div>
            </section>

            {/* About Section */}
            <section
                ref={aboutRef}
                className={`about-section ${isAboutVisible ? 'visible' : ''}`}
                style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}
            >
                <div className="about-container" style={{ width: '100%', maxWidth: '1200px', margin: '0 auto', padding: '0 32px' }}>
                    <div className="about-header" style={{ textAlign: 'center', width: '100%' }}>
                        <span className="section-tag">Our Mission</span>
                        <h2 className="about-title" style={{ textAlign: 'center', width: '100%' }}>
                            From Burned Land to <span className="text-gradient">Green Future</span>
                        </h2>
                        <p className="about-lead" style={{ textAlign: 'center', margin: '0 auto' }}>
                            EcoRevive uses advanced AI and satellite imagery to transform ecosystem restoration
                            from guesswork into precision science.
                        </p>
                    </div>

                    <div className="features-grid">
                        <div className="feature-card">
                            <div className="feature-icon">
                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <circle cx="12" cy="12" r="10" />
                                    <path d="M2 12h4M18 12h4M12 2v4M12 18v4" />
                                    <circle cx="12" cy="12" r="3" />
                                </svg>
                            </div>
                            <h3 className="feature-title">Satellite Analysis</h3>
                            <p className="feature-desc">
                                10-band Sentinel-2 imagery processed by our deep learning model to detect
                                degradation with unmatched accuracy.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">
                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path d="M12 2a10 10 0 0 1 10 10c0 5.523-4.477 10-10 10S2 17.523 2 12" />
                                    <path d="M12 6v6l4 2" />
                                    <circle cx="12" cy="12" r="2" />
                                </svg>
                            </div>
                            <h3 className="feature-title">AI Reasoning</h3>
                            <p className="feature-desc">
                                Gemini-powered intelligence generates species recommendations, safety protocols,
                                and restoration timelines.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">
                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path d="M3 3v18h18" />
                                    <path d="M7 16l4-4 4 4 5-6" />
                                </svg>
                            </div>
                            <h3 className="feature-title">Actionable Reports</h3>
                            <p className="feature-desc">
                                Professional-grade outputs with legal compliance, cost-benefit analysis,
                                and monitoring frameworks.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">
                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path d="M12 22c4-4 8-7.5 8-12a8 8 0 1 0-16 0c0 4.5 4 8 8 12z" />
                                    <path d="M12 6v6M9 10l3 3 3-3" />
                                </svg>
                            </div>
                            <h3 className="feature-title">Hope Visualizer</h3>
                            <p className="feature-desc">
                                See the future of your restoration site with AI-generated recovery forecasts
                                spanning 15+ years.
                            </p>
                        </div>
                    </div>

                    {/* CTA */}
                    <div className="about-cta" style={{ textAlign: 'center', width: '100%' }}>
                        <button className="btn btn-primary btn-lg" onClick={handleExploreClick}>
                            <span>Start Exploring</span>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M5 12h14M12 5l7 7-7 7" />
                            </svg>
                        </button>
                        <p className="cta-note" style={{ textAlign: 'center' }}>Select any location on Earth to begin analysis</p>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="landing-footer">
                <div className="footer-content">
                    <span className="footer-logo">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
                            <path d="M12 8v8M8 12h8" />
                        </svg>
                        EcoRevive
                    </span>
                    <span className="footer-text">Built for the Gemini Hackathon 2026</span>
                </div>
            </footer>

            {/* User Type Selection Modal */}
            {showModal && (
                <div className="modal-overlay" onClick={handleCloseModal}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <button className="modal-close" onClick={handleCloseModal}>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M18 6L6 18M6 6l12 12" />
                            </svg>
                        </button>

                        <div className="modal-header">
                            <h2>Welcome to EcoRevive</h2>
                            <p>I will be using the results of this app for...</p>
                        </div>

                        <div className="modal-options">
                            <button
                                className={`user-type-option ${selectedUserType === 'personal' ? 'selected' : ''}`}
                                onClick={() => handleUserTypeSelect('personal')}
                            >
                                <div className="option-icon">
                                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                        <circle cx="12" cy="8" r="4" />
                                        <path d="M4 20c0-4 4-6 8-6s8 2 8 6" />
                                        <path d="M15 5l2-2m0 0l2 2m-2-2v4" opacity="0.5" />
                                    </svg>
                                </div>
                                <div className="option-text">
                                    <h3>Personal Use</h3>
                                    <p>Exploring for curiosity, education, or small-scale projects</p>
                                </div>
                            </button>

                            <button
                                className={`user-type-option ${selectedUserType === 'professional' ? 'selected' : ''}`}
                                onClick={() => handleUserTypeSelect('professional')}
                            >
                                <div className="option-icon">
                                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                        <rect x="3" y="7" width="18" height="14" rx="2" />
                                        <path d="M8 7V5a2 2 0 012-2h4a2 2 0 012 2v2" />
                                        <path d="M12 12v4" />
                                        <path d="M8 14h8" />
                                    </svg>
                                </div>
                                <div className="option-text">
                                    <h3>Professional Organization</h3>
                                    <p>Government agencies, NGOs, or enterprise restoration projects</p>
                                </div>
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
