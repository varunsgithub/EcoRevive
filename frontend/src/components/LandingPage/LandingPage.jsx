import { useState, useEffect, useRef } from 'react'
import * as THREE from 'three'
import './LandingPage.css'

// Animated 3D Globe Component
function AnimatedGlobe() {
    const containerRef = useRef(null)
    const sceneRef = useRef(null)

    useEffect(() => {
        if (!containerRef.current || sceneRef.current) return

        const container = containerRef.current
        const width = container.clientWidth
        const height = container.clientHeight

        // Scene setup
        const scene = new THREE.Scene()
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000)
        camera.position.z = 2.5

        const renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        })
        renderer.setSize(width, height)
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
        container.appendChild(renderer.domElement)

        // Create dotted sphere (globe)
        const globeGeometry = new THREE.SphereGeometry(1, 64, 64)

        // Create points on sphere surface
        const positions = []
        const colors = []
        const color = new THREE.Color()

        for (let i = 0; i < 3000; i++) {
            const phi = Math.acos(-1 + (2 * i) / 3000)
            const theta = Math.sqrt(3000 * Math.PI) * phi

            const x = Math.cos(theta) * Math.sin(phi)
            const y = Math.sin(theta) * Math.sin(phi)
            const z = Math.cos(phi)

            positions.push(x, y, z)

            // Color variation based on position
            const intensity = 0.3 + Math.random() * 0.7
            color.setRGB(0, intensity * 0.83, intensity * 0.67)
            colors.push(color.r, color.g, color.b)
        }

        const pointsGeometry = new THREE.BufferGeometry()
        pointsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
        pointsGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))

        const pointsMaterial = new THREE.PointsMaterial({
            size: 0.015,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            sizeAttenuation: true
        })

        const globe = new THREE.Points(pointsGeometry, pointsMaterial)
        scene.add(globe)

        // Add wireframe sphere for structure
        const wireframeMaterial = new THREE.MeshBasicMaterial({
            color: 0x00d4aa,
            wireframe: true,
            transparent: true,
            opacity: 0.05
        })
        const wireframeSphere = new THREE.Mesh(globeGeometry, wireframeMaterial)
        scene.add(wireframeSphere)

        // Create floating particles
        const particlesGeometry = new THREE.BufferGeometry()
        const particlePositions = []
        const particleCount = 200

        for (let i = 0; i < particleCount; i++) {
            const radius = 1.5 + Math.random() * 1.5
            const theta = Math.random() * Math.PI * 2
            const phi = Math.random() * Math.PI

            particlePositions.push(
                radius * Math.sin(phi) * Math.cos(theta),
                radius * Math.sin(phi) * Math.sin(theta),
                radius * Math.cos(phi)
            )
        }

        particlesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(particlePositions, 3))

        const particlesMaterial = new THREE.PointsMaterial({
            size: 0.02,
            color: 0x00d4aa,
            transparent: true,
            opacity: 0.5,
            sizeAttenuation: true
        })

        const particles = new THREE.Points(particlesGeometry, particlesMaterial)
        scene.add(particles)

        // Add glow ring
        const ringGeometry = new THREE.TorusGeometry(1.2, 0.01, 2, 100)
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: 0x00d4aa,
            transparent: true,
            opacity: 0.3
        })
        const ring = new THREE.Mesh(ringGeometry, ringMaterial)
        ring.rotation.x = Math.PI / 2
        scene.add(ring)

        // Animation
        let animationId
        const animate = () => {
            animationId = requestAnimationFrame(animate)

            globe.rotation.y += 0.002
            wireframeSphere.rotation.y += 0.002
            particles.rotation.y -= 0.001
            particles.rotation.x += 0.0005
            ring.rotation.z += 0.003

            renderer.render(scene, camera)
        }
        animate()

        // Handle resize
        const handleResize = () => {
            const newWidth = container.clientWidth
            const newHeight = container.clientHeight
            camera.aspect = newWidth / newHeight
            camera.updateProjectionMatrix()
            renderer.setSize(newWidth, newHeight)
        }
        window.addEventListener('resize', handleResize)

        sceneRef.current = { scene, renderer, animationId }

        return () => {
            window.removeEventListener('resize', handleResize)
            cancelAnimationFrame(animationId)
            renderer.dispose()
            container.removeChild(renderer.domElement)
            sceneRef.current = null
        }
    }, [])

    return <div ref={containerRef} className="globe-canvas" />
}

export default function LandingPage({ onEnterGlobe, isTransitioning }) {
    const [showModal, setShowModal] = useState(false)
    const [selectedUserType, setSelectedUserType] = useState(null)

    // Scroll reveal effect
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('revealed')
                    }
                })
            },
            { threshold: 0.1 }
        )

        document.querySelectorAll('.scroll-reveal').forEach(el => {
            observer.observe(el)
        })

        return () => observer.disconnect()
    }, [])

    const handleStartClick = () => setShowModal(true)
    const handleCloseModal = () => {
        setShowModal(false)
        setSelectedUserType(null)
    }

    const handleUserTypeSelect = (userType) => {
        setSelectedUserType(userType)
        setTimeout(() => {
            setShowModal(false)
            onEnterGlobe(userType)
        }, 200)
    }

    return (
        <div className={`landing ${isTransitioning ? 'fade-out' : ''}`}>
            {/* Ambient Background */}
            <div className="ambient-bg">
                <div className="gradient-orb orb-1" />
                <div className="gradient-orb orb-2" />
            </div>

            {/* Navigation */}
            <nav className="nav">
                <div className="logo">
                    <div className="logo-mark">
                        <svg viewBox="0 0 24 24" fill="none">
                            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5"/>
                            <path d="M12 2C12 2 8 6 8 12s4 10 4 10" stroke="currentColor" strokeWidth="1.5"/>
                            <path d="M12 2C12 2 16 6 16 12s-4 10-4 10" stroke="currentColor" strokeWidth="1.5"/>
                            <path d="M2 12h20" stroke="currentColor" strokeWidth="1.5"/>
                        </svg>
                    </div>
                    <span>EcoRevive</span>
                </div>
                <button className="nav-cta" onClick={handleStartClick}>
                    Launch Platform
                </button>
            </nav>

            {/* Hero Section */}
            <section className="hero">
                <div className="hero-content">
                    <p className="hero-eyebrow animate-fade-up delay-1">
                        <span className="eyebrow-dot" />
                        AI-Powered Restoration Planning
                    </p>
                    <h1 className="animate-fade-up delay-2">
                        Turn Burned Land Into<br />
                        <span className="gradient-text">Thriving Ecosystems</span>
                    </h1>
                    <p className="hero-description animate-fade-up delay-3">
                        Analyze wildfire damage with satellite imagery, calculate carbon
                        sequestration potential, and generate professional restoration plans.
                    </p>
                    <div className="hero-actions animate-fade-up delay-4">
                        <button className="btn-glow" onClick={handleStartClick}>
                            <span>Start Analysis</span>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M5 12h14m-7-7l7 7-7 7" />
                            </svg>
                        </button>
                        <a href="#features" className="btn-ghost">
                            Learn More
                        </a>
                    </div>
                </div>
                <div className="hero-globe">
                    <AnimatedGlobe />
                    <div className="globe-glow" />
                </div>
            </section>

            {/* Stats Bar */}
            <section className="stats-section">
                <div className="stats-bar glass-card scroll-reveal">
                    <div className="stat">
                        <span className="stat-num">10m</span>
                        <span className="stat-label">Resolution</span>
                    </div>
                    <div className="stat-divider" />
                    <div className="stat">
                        <span className="stat-num">Sentinel-2</span>
                        <span className="stat-label">Satellite Data</span>
                    </div>
                    <div className="stat-divider" />
                    <div className="stat">
                        <span className="stat-num">Gemini</span>
                        <span className="stat-label">AI Analysis</span>
                    </div>
                    <div className="stat-divider" />
                    <div className="stat">
                        <span className="stat-num">IPCC</span>
                        <span className="stat-label">Carbon Method</span>
                    </div>
                </div>
            </section>

            {/* Features */}
            <section className="section" id="features">
                <div className="section-header scroll-reveal">
                    <h2>Platform Capabilities</h2>
                    <p>Everything you need for restoration planning</p>
                </div>
                <div className="features-grid">
                    <div className="feature-card glass-card scroll-reveal">
                        <div className="feature-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <rect x="3" y="3" width="18" height="18" rx="2" />
                                <path d="M3 9h18M9 21V9" />
                            </svg>
                        </div>
                        <h3>Burn Severity Mapping</h3>
                        <p>Deep learning model provides pixel-level severity classification at 10m resolution</p>
                    </div>
                    <div className="feature-card glass-card scroll-reveal">
                        <div className="feature-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                                <path d="M2 17l10 5 10-5M2 12l10 5 10-5" />
                            </svg>
                        </div>
                        <h3>Carbon Accounting</h3>
                        <p>IPCC Tier 2 methodology calculates sequestration and carbon credit eligibility</p>
                    </div>
                    <div className="feature-card glass-card scroll-reveal">
                        <div className="feature-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                            </svg>
                        </div>
                        <h3>AI Assistant</h3>
                        <p>Ask questions about safety, species selection, and site-specific recommendations</p>
                    </div>
                </div>
            </section>

            {/* How It Works */}
            <section className="section section-dark">
                <div className="section-header scroll-reveal">
                    <h2>How It Works</h2>
                    <p>From selection to restoration plan in minutes</p>
                </div>
                <div className="steps-container scroll-reveal">
                    <div className="step-card">
                        <div className="step-num">01</div>
                        <h3>Select Location</h3>
                        <p>Use the interactive globe to select any burn-affected area</p>
                    </div>
                    <div className="step-arrow">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M5 12h14m-7-7l7 7-7 7" />
                        </svg>
                    </div>
                    <div className="step-card">
                        <div className="step-num">02</div>
                        <h3>AI Analysis</h3>
                        <p>Our model processes Sentinel-2 imagery to map severity</p>
                    </div>
                    <div className="step-arrow">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M5 12h14m-7-7l7 7-7 7" />
                        </svg>
                    </div>
                    <div className="step-card">
                        <div className="step-num">03</div>
                        <h3>Get Results</h3>
                        <p>Receive severity maps, carbon estimates, and recommendations</p>
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className="section-cta">
                <div className="cta-content scroll-reveal">
                    <h2>Ready to restore?</h2>
                    <p>Select a location and get your analysis in seconds</p>
                    <button className="btn-glow btn-large" onClick={handleStartClick}>
                        <span>Launch Platform</span>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M5 12h14m-7-7l7 7-7 7" />
                        </svg>
                    </button>
                </div>
            </section>

            {/* Footer */}
            <footer className="footer">
                <div className="footer-content">
                    <div className="footer-brand">
                        <div className="logo-mark-small">
                            <svg viewBox="0 0 24 24" fill="none">
                                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5"/>
                            </svg>
                        </div>
                        <span>EcoRevive</span>
                    </div>
                    <div className="footer-links">
                        <span>Gemini Hackathon 2026</span>
                    </div>
                </div>
            </footer>

            {/* Modal */}
            {showModal && (
                <div className="modal-backdrop" onClick={handleCloseModal}>
                    <div className="modal glass-card" onClick={e => e.stopPropagation()}>
                        <button className="modal-close" onClick={handleCloseModal}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M18 6L6 18M6 6l12 12" />
                            </svg>
                        </button>

                        <h2>Select Your Use Case</h2>
                        <p>We'll tailor the analysis accordingly</p>

                        <div className="modal-options">
                            <button
                                className={`option ${selectedUserType === 'personal' ? 'selected' : ''}`}
                                onClick={() => handleUserTypeSelect('personal')}
                            >
                                <div className="option-icon">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                        <circle cx="12" cy="8" r="4" />
                                        <path d="M20 21a8 8 0 10-16 0" />
                                    </svg>
                                </div>
                                <div className="option-content">
                                    <h3>Personal</h3>
                                    <p>Community organizing, education, research</p>
                                </div>
                                <div className="option-check">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M20 6L9 17l-5-5" />
                                    </svg>
                                </div>
                            </button>

                            <button
                                className={`option ${selectedUserType === 'professional' ? 'selected' : ''}`}
                                onClick={() => handleUserTypeSelect('professional')}
                            >
                                <div className="option-icon">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                        <path d="M3 21h18M5 21V7l8-4 8 4v14" />
                                        <path d="M9 21v-4h6v4M10 9h4M10 13h4" />
                                    </svg>
                                </div>
                                <div className="option-content">
                                    <h3>Professional</h3>
                                    <p>Government, NGO, enterprise projects</p>
                                </div>
                                <div className="option-check">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M20 6L9 17l-5-5" />
                                    </svg>
                                </div>
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
