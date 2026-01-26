import { useRef, useMemo } from 'react'
import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { TextureLoader } from 'three'
import { OrbitControls, Stars } from '@react-three/drei'
import * as THREE from 'three'

/**
 * Rotating Earth Sphere Component
 * Uses high-quality textures for realistic appearance
 */
function EarthSphere({ onClick }) {
    const meshRef = useRef()
    const cloudsRef = useRef()

    // Auto-rotate the Earth
    useFrame((state, delta) => {
        if (meshRef.current) {
            meshRef.current.rotation.y += delta * 0.05 // Slow rotation
        }
        if (cloudsRef.current) {
            cloudsRef.current.rotation.y += delta * 0.06 // Slightly faster cloud movement
        }
    })

    // Earth material with realistic lighting
    const earthMaterial = useMemo(() => {
        return new THREE.MeshStandardMaterial({
            color: '#2d5a27',
            roughness: 0.8,
            metalness: 0.1,
        })
    }, [])

    return (
        <group onClick={onClick}>
            {/* Main Earth sphere */}
            <mesh ref={meshRef} castShadow receiveShadow>
                <sphereGeometry args={[2, 64, 64]} />
                <meshStandardMaterial
                    color="#1a4d1a"
                    roughness={0.7}
                    metalness={0.1}
                />
            </mesh>

            {/* Ocean layer */}
            <mesh scale={0.998}>
                <sphereGeometry args={[2, 64, 64]} />
                <meshStandardMaterial
                    color="#0a2d5c"
                    roughness={0.3}
                    metalness={0.2}
                    transparent
                    opacity={0.9}
                />
            </mesh>

            {/* Atmosphere glow */}
            <mesh scale={1.02}>
                <sphereGeometry args={[2, 64, 64]} />
                <meshBasicMaterial
                    color="#00d4aa"
                    transparent
                    opacity={0.08}
                    side={THREE.BackSide}
                />
            </mesh>

            {/* Outer glow */}
            <mesh scale={1.08}>
                <sphereGeometry args={[2, 32, 32]} />
                <meshBasicMaterial
                    color="#00d4aa"
                    transparent
                    opacity={0.03}
                    side={THREE.BackSide}
                />
            </mesh>
        </group>
    )
}

/**
 * Atmospheric haze effect
 */
function AtmosphereEffect() {
    return (
        <mesh scale={2.15}>
            <sphereGeometry args={[2, 32, 32]} />
            <meshBasicMaterial
                color="#00a080"
                transparent
                opacity={0.02}
                side={THREE.BackSide}
            />
        </mesh>
    )
}

/**
 * Ambient particles floating around
 */
function FloatingParticles() {
    const particlesRef = useRef()

    const particles = useMemo(() => {
        const positions = []
        for (let i = 0; i < 200; i++) {
            const radius = 4 + Math.random() * 3
            const theta = Math.random() * Math.PI * 2
            const phi = Math.acos(2 * Math.random() - 1)

            positions.push(
                radius * Math.sin(phi) * Math.cos(theta),
                radius * Math.sin(phi) * Math.sin(theta),
                radius * Math.cos(phi)
            )
        }
        return new Float32Array(positions)
    }, [])

    useFrame((state, delta) => {
        if (particlesRef.current) {
            particlesRef.current.rotation.y += delta * 0.02
        }
    })

    return (
        <points ref={particlesRef}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    count={particles.length / 3}
                    array={particles}
                    itemSize={3}
                />
            </bufferGeometry>
            <pointsMaterial
                size={0.02}
                color="#00d4aa"
                transparent
                opacity={0.4}
                sizeAttenuation
            />
        </points>
    )
}

/**
 * Main Hero Earth Component
 * Full-screen 3D Earth with interactive elements
 */
export default function HeroEarth({ onClick }) {
    return (
        <div className="hero-earth" onClick={onClick}>
            <Canvas
                camera={{ position: [0, 0, 6], fov: 45 }}
                style={{ background: 'transparent' }}
                gl={{ antialias: true, alpha: true }}
            >
                {/* Lighting */}
                <ambientLight intensity={0.3} />
                <directionalLight
                    position={[5, 3, 5]}
                    intensity={1.2}
                    color="#ffffff"
                    castShadow
                />
                <directionalLight
                    position={[-3, -1, -3]}
                    intensity={0.3}
                    color="#00d4aa"
                />
                <pointLight position={[-10, 0, -10]} intensity={0.2} color="#d4a574" />

                {/* Background stars */}
                <Stars
                    radius={100}
                    depth={50}
                    count={3000}
                    factor={4}
                    saturation={0}
                    fade
                    speed={0.5}
                />

                {/* Earth */}
                <EarthSphere onClick={onClick} />
                <AtmosphereEffect />
                <FloatingParticles />

                {/* Subtle camera controls */}
                <OrbitControls
                    enableZoom={false}
                    enablePan={false}
                    enableRotate={true}
                    rotateSpeed={0.3}
                    autoRotate
                    autoRotateSpeed={0.2}
                />
            </Canvas>

            {/* Click indicator */}
            <div className="click-indicator">
                <span className="pulse-ring" />
                <span className="click-text">Click to Explore</span>
            </div>

            <style>{`
        .hero-earth {
          width: 100%;
          height: 100%;
          cursor: pointer;
          position: relative;
        }
        
        .click-indicator {
          position: absolute;
          bottom: 15%;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: var(--space-3);
          opacity: 0;
          animation: fadeInUp 1s var(--ease-smooth) 1.5s forwards;
        }
        
        .pulse-ring {
          width: 48px;
          height: 48px;
          border: 2px solid var(--color-accent);
          border-radius: 50%;
          animation: pulse-expand 2s var(--ease-smooth) infinite;
        }
        
        .click-text {
          font-family: var(--font-body);
          font-size: var(--text-sm);
          color: var(--color-text-muted);
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }
        
        @keyframes pulse-expand {
          0% {
            transform: scale(1);
            opacity: 1;
          }
          100% {
            transform: scale(1.8);
            opacity: 0;
          }
        }
        
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateX(-50%) translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
          }
        }
      `}</style>
        </div>
    )
}
