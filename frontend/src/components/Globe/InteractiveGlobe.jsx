import { useState, useEffect, useRef, useCallback } from 'react'
import './InteractiveGlobe.css'

// Cesium imports - will be loaded dynamically
let Viewer, Cartesian3, Cartographic, CesiumMath, Rectangle, Color, ScreenSpaceEventHandler, ScreenSpaceEventType, CallbackProperty

/**
 * Interactive Globe Component
 * CesiumJS-powered satellite imagery explorer with drag-to-draw area selection
 */
export default function InteractiveGlobe({ onBack, isTransitioning }) {
    const cesiumContainerRef = useRef(null)
    const viewerRef = useRef(null)
    const handlerRef = useRef(null)
    const [isLoading, setIsLoading] = useState(true)
    const [cesiumLoaded, setCesiumLoaded] = useState(false)
    const [searchQuery, setSearchQuery] = useState('')
    const [currentAltitude, setCurrentAltitude] = useState(0)
    const [isZoomedIn, setIsZoomedIn] = useState(false) // Based on altitude threshold
    const [selectionModeEnabled, setSelectionModeEnabled] = useState(false) // User toggle
    const [selectedBounds, setSelectedBounds] = useState(null)
    const [isDrawing, setIsDrawing] = useState(false)
    const [drawStart, setDrawStart] = useState(null)
    const [searchResults, setSearchResults] = useState([])
    const [showSearchResults, setShowSearchResults] = useState(false)
    const [selectionConfirmed, setSelectionConfirmed] = useState(false)

    // Analysis state
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [analysisResults, setAnalysisResults] = useState(null)
    const [analysisError, setAnalysisError] = useState(null)

    // Coordinate input state
    const [showCoordinateInput, setShowCoordinateInput] = useState(false)
    const [coordLat, setCoordLat] = useState('')
    const [coordLon, setCoordLon] = useState('')

    // Altitude threshold for enabling selection mode button (in meters)
    const SELECTION_THRESHOLD = 50000 // 50km

    // Load Cesium dynamically
    useEffect(() => {
        const loadCesium = async () => {
            try {
                const cesium = await import('cesium')
                Viewer = cesium.Viewer
                Cartesian3 = cesium.Cartesian3
                Cartographic = cesium.Cartographic
                CesiumMath = cesium.Math
                Rectangle = cesium.Rectangle
                Color = cesium.Color
                ScreenSpaceEventHandler = cesium.ScreenSpaceEventHandler
                ScreenSpaceEventType = cesium.ScreenSpaceEventType
                CallbackProperty = cesium.CallbackProperty

                // Set Cesium Ion token (using default for demo)
                cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYWE1OWUxNy1mMWZiLTQzYjYtYTQ0OS1kMWFjYmFkNjc5YzciLCJpZCI6NTc3MzMsImlhdCI6MTYyNzI0OTI2NX0.XcKpgANiY19MC4bdFUXMVEBToBmqS8kuYpUlxJHYZxk'

                setCesiumLoaded(true)
            } catch (error) {
                console.error('Failed to load Cesium:', error)
            }
        }
        loadCesium()
    }, [])

    // Initialize Cesium viewer
    useEffect(() => {
        if (!cesiumLoaded || !cesiumContainerRef.current || viewerRef.current) return

        const initViewer = async () => {
            try {
                const cesium = await import('cesium')
                const OpenStreetMapImageryProvider = cesium.OpenStreetMapImageryProvider

                const viewer = new Viewer(cesiumContainerRef.current, {
                    animation: false,
                    baseLayerPicker: false,
                    fullscreenButton: false,
                    vrButton: false,
                    geocoder: false,
                    homeButton: false,
                    infoBox: false,
                    sceneModePicker: false,
                    selectionIndicator: false,
                    timeline: false,
                    navigationHelpButton: false,
                    navigationInstructionsInitiallyVisible: false,
                    skyBox: false,
                    skyAtmosphere: false,
                    baseLayer: false,
                    contextOptions: {
                        webgl: {
                            alpha: true
                        }
                    }
                })

                // Add OpenStreetMap imagery layer
                const osmProvider = new OpenStreetMapImageryProvider({
                    url: 'https://tile.openstreetmap.org/'
                })
                viewer.imageryLayers.addImageryProvider(osmProvider)

                // Remove Cesium credits
                viewer.cesiumWidget.creditContainer.style.display = 'none'

                // Set initial camera position
                viewer.camera.setView({
                    destination: Cartesian3.fromDegrees(-119.5, 37.5, 5000000),
                    orientation: {
                        heading: 0,
                        pitch: CesiumMath.toRadians(-90),
                        roll: 0
                    }
                })

                viewerRef.current = viewer
                setIsLoading(false)

                // Track camera altitude
                viewer.camera.changed.addEventListener(() => {
                    const cartographic = Cartographic.fromCartesian(viewer.camera.position)
                    const altitude = cartographic.height
                    setCurrentAltitude(altitude)

                    // Track if user is zoomed in enough for selection (but don't auto-enable)
                    if (altitude < SELECTION_THRESHOLD) {
                        setIsZoomedIn(true)
                    } else {
                        setIsZoomedIn(false)
                        // Disable selection mode when zooming out
                        if (!selectedBounds) {
                            setSelectionModeEnabled(false)
                        }
                    }
                })
            } catch (error) {
                console.error('Failed to initialize Cesium viewer:', error)
                setIsLoading(false)
            }
        }

        initViewer()

        return () => {
            if (handlerRef.current) {
                handlerRef.current.destroy()
            }
            if (viewerRef.current) {
                viewerRef.current.destroy()
                viewerRef.current = null
            }
        }
    }, [cesiumLoaded])

    // Disable camera controls when in selection mode (user-enabled)
    useEffect(() => {
        if (!viewerRef.current) return

        const viewer = viewerRef.current
        const controller = viewer.scene.screenSpaceCameraController

        if (selectionModeEnabled) {
            // Disable left-drag rotation so we can draw selection
            controller.enableRotate = false
            controller.enableTranslate = false
            controller.enableTilt = false
            // Keep zoom enabled
            controller.enableZoom = true
        } else {
            // Re-enable all controls
            controller.enableRotate = true
            controller.enableTranslate = true
            controller.enableTilt = true
            controller.enableZoom = true
        }
    }, [selectionModeEnabled])

    // Setup mouse handlers for drawing selection
    useEffect(() => {
        if (!viewerRef.current || !selectionModeEnabled || handlerRef.current) return

        const viewer = viewerRef.current
        const handler = new ScreenSpaceEventHandler(viewer.scene.canvas)
        handlerRef.current = handler

        let startPosition = null
        let currentRectangle = null
        let rectangleEntity = null

        // Mouse down - start drawing
        handler.setInputAction((event) => {
            const cartesian = viewer.camera.pickEllipsoid(event.position, viewer.scene.globe.ellipsoid)
            if (cartesian) {
                const cartographic = Cartographic.fromCartesian(cartesian)
                startPosition = {
                    lon: CesiumMath.toDegrees(cartographic.longitude),
                    lat: CesiumMath.toDegrees(cartographic.latitude)
                }

                // Initialize rectangle with same start/end
                currentRectangle = Rectangle.fromDegrees(
                    startPosition.lon,
                    startPosition.lat,
                    startPosition.lon,
                    startPosition.lat
                )

                setDrawStart(startPosition)
                setIsDrawing(true)

                // Clear previous selection
                viewer.entities.removeAll()
                setSelectedBounds(null)
                setSelectionConfirmed(false)

                // Create entity with CallbackProperty for smooth updates
                rectangleEntity = viewer.entities.add({
                    rectangle: {
                        coordinates: new CallbackProperty(() => currentRectangle, false),
                        material: Color.fromCssColorString('#00d4aa').withAlpha(0.3),
                        outline: true,
                        outlineColor: Color.fromCssColorString('#00d4aa'),
                        outlineWidth: 2
                    }
                })
            }
        }, ScreenSpaceEventType.LEFT_DOWN)

        // Mouse move - update rectangle (no entity recreation!)
        handler.setInputAction((event) => {
            if (!startPosition || !currentRectangle) return

            const cartesian = viewer.camera.pickEllipsoid(event.endPosition, viewer.scene.globe.ellipsoid)
            if (cartesian) {
                const cartographic = Cartographic.fromCartesian(cartesian)
                const endLon = CesiumMath.toDegrees(cartographic.longitude)
                const endLat = CesiumMath.toDegrees(cartographic.latitude)

                // Update the existing rectangle - CallbackProperty will pick this up
                currentRectangle = Rectangle.fromDegrees(
                    Math.min(startPosition.lon, endLon),
                    Math.min(startPosition.lat, endLat),
                    Math.max(startPosition.lon, endLon),
                    Math.max(startPosition.lat, endLat)
                )
            }
        }, ScreenSpaceEventType.MOUSE_MOVE)

        // Mouse up - finish drawing
        handler.setInputAction(() => {
            if (currentRectangle && startPosition) {
                const west = CesiumMath.toDegrees(currentRectangle.west)
                const south = CesiumMath.toDegrees(currentRectangle.south)
                const east = CesiumMath.toDegrees(currentRectangle.east)
                const north = CesiumMath.toDegrees(currentRectangle.north)

                const width = Math.abs(east - west)
                const height = Math.abs(north - south)

                if (width > 0.001 && height > 0.001) {
                    setSelectedBounds({ west, south, east, north })

                    // Convert to static rectangle (no longer callback)
                    if (rectangleEntity) {
                        viewer.entities.remove(rectangleEntity)
                        viewer.entities.add({
                            rectangle: {
                                coordinates: Rectangle.fromDegrees(west, south, east, north),
                                material: Color.fromCssColorString('#00d4aa').withAlpha(0.3),
                                outline: true,
                                outlineColor: Color.fromCssColorString('#00d4aa'),
                                outlineWidth: 2
                            }
                        })
                    }
                } else {
                    // Too small, clear
                    viewer.entities.removeAll()
                }
            }
            startPosition = null
            currentRectangle = null
            rectangleEntity = null
            setIsDrawing(false)
            setDrawStart(null)
        }, ScreenSpaceEventType.LEFT_UP)

        return () => {
            handler.destroy()
            handlerRef.current = null
        }
    }, [selectionModeEnabled, viewerRef.current])

    // Search location
    const handleSearch = useCallback(async () => {
        if (!searchQuery.trim()) return

        try {
            const response = await fetch(
                `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=5`
            )
            const results = await response.json()
            setSearchResults(results)
            setShowSearchResults(true)
        } catch (error) {
            console.error('Search failed:', error)
        }
    }, [searchQuery])

    // Fly to location
    const flyToLocation = useCallback((lon, lat, name) => {
        if (!viewerRef.current) return

        viewerRef.current.camera.flyTo({
            destination: Cartesian3.fromDegrees(parseFloat(lon), parseFloat(lat), 30000),
            duration: 2,
            orientation: {
                heading: 0,
                pitch: CesiumMath.toRadians(-60),
                roll: 0
            }
        })

        setSearchQuery(name)
        setShowSearchResults(false)
    }, [])

    // Handle key press for search
    const handleKeyPress = useCallback((e) => {
        if (e.key === 'Enter') {
            handleSearch()
        }
    }, [handleSearch])

    // Fly to coordinates
    const handleFlyToCoordinates = useCallback(() => {
        if (!viewerRef.current) return

        // Parse latitude
        let lat = parseFloat(coordLat)
        if (coordLat.toLowerCase().includes('s')) {
            lat = -Math.abs(lat)
        }

        // Parse longitude  
        let lon = parseFloat(coordLon)
        if (coordLon.toLowerCase().includes('w')) {
            lon = -Math.abs(lon)
        }

        // Validate
        if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
            alert('Invalid coordinates. Latitude must be -90 to 90, Longitude -180 to 180.')
            return
        }

        console.log(`Flying to coordinates: ${lat}, ${lon}`)

        viewerRef.current.camera.flyTo({
            destination: Cartesian3.fromDegrees(lon, lat, 30000),
            duration: 2,
            orientation: {
                heading: 0,
                pitch: CesiumMath.toRadians(-60),
                roll: 0
            }
        })

        setShowCoordinateInput(false)
    }, [coordLat, coordLon])

    // Clear selection
    const handleClearSelection = useCallback(() => {
        if (viewerRef.current) {
            viewerRef.current.entities.removeAll()
        }
        setSelectedBounds(null)
        setSelectionConfirmed(false)
        setIsAnalyzing(false)
        setAnalysisResults(null)
        setAnalysisError(null)
    }, [])

    // Confirm selection and trigger analysis
    const handleConfirmSelection = useCallback(async () => {
        if (!selectedBounds) return

        console.log('Selection confirmed:', selectedBounds)
        setSelectionConfirmed(true)
        setIsAnalyzing(true)
        setAnalysisError(null)
        setAnalysisResults(null)

        try {
            const response = await fetch('http://localhost:8000/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    west: selectedBounds.west,
                    south: selectedBounds.south,
                    east: selectedBounds.east,
                    north: selectedBounds.north,
                    user_type: 'personal'  // TODO: get from context
                }),
            })

            const data = await response.json()

            if (data.success) {
                setAnalysisResults(data)
                console.log('Analysis complete:', data)
            } else {
                setAnalysisError(data.error || 'Analysis failed')
            }
        } catch (error) {
            console.error('Analysis error:', error)
            setAnalysisError(`Failed to connect to analysis server: ${error.message}`)
        } finally {
            setIsAnalyzing(false)
        }
    }, [selectedBounds])

    // Format altitude for display
    const formatAltitude = (meters) => {
        if (meters > 1000000) {
            return `${(meters / 1000000).toFixed(1)}M km`
        } else if (meters > 1000) {
            return `${(meters / 1000).toFixed(1)} km`
        }
        return `${meters.toFixed(0)} m`
    }

    // Calculate area in km²
    const calculateArea = (bounds) => {
        if (!bounds) return 0
        const latMid = (bounds.north + bounds.south) / 2
        const kmPerDegLon = 111.32 * Math.cos(latMid * Math.PI / 180)
        const kmPerDegLat = 110.574
        const width = Math.abs(bounds.east - bounds.west) * kmPerDegLon
        const height = Math.abs(bounds.north - bounds.south) * kmPerDegLat
        return (width * height).toFixed(1)
    }

    return (
        <div className={`interactive-globe ${isTransitioning ? 'transitioning' : ''}`}>
            {/* Loading overlay */}
            {isLoading && (
                <div className="globe-loading">
                    <div className="loading-spinner" />
                    <p>Loading satellite imagery...</p>
                </div>
            )}

            {/* Cesium container */}
            <div ref={cesiumContainerRef} className="cesium-container" />

            {/* Top bar */}
            <div className="globe-topbar">
                <button className="back-button" onClick={onBack}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M19 12H5M12 19l-7-7 7-7" />
                    </svg>
                    <span>Back</span>
                </button>

                <div className="search-container">
                    <input
                        type="text"
                        className="search-input"
                        placeholder="Search location..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyPress={handleKeyPress}
                        onFocus={() => searchResults.length > 0 && setShowSearchResults(true)}
                    />
                    <button className="search-button" onClick={handleSearch}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="11" cy="11" r="8" />
                            <path d="M21 21l-4.35-4.35" />
                        </svg>
                    </button>

                    {/* Search results dropdown */}
                    {showSearchResults && searchResults.length > 0 && (
                        <div className="search-results">
                            {searchResults.map((result, index) => (
                                <button
                                    key={index}
                                    className="search-result-item"
                                    onClick={() => flyToLocation(result.lon, result.lat, result.display_name)}
                                >
                                    <span className="result-icon">
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z" />
                                            <circle cx="12" cy="10" r="3" />
                                        </svg>
                                    </span>
                                    <span className="result-text">{result.display_name}</span>
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <div className="altitude-display">
                    <span className="altitude-label">Altitude</span>
                    <span className="altitude-value">{formatAltitude(currentAltitude)}</span>
                </div>

                {/* Selection Mode Toggle - only visible when zoomed in */}
                {isZoomedIn && !selectedBounds && (
                    <button
                        className={`selection-toggle-button ${selectionModeEnabled ? 'active' : ''}`}
                        onClick={() => setSelectionModeEnabled(!selectionModeEnabled)}
                        title={selectionModeEnabled ? "Exit selection mode" : "Enter selection mode to draw area"}
                    >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" strokeDasharray={selectionModeEnabled ? "0" : "4 2"} />
                            {selectionModeEnabled && <path d="M9 12l2 2 4-4" />}
                        </svg>
                        <span>{selectionModeEnabled ? 'Drawing' : 'Select'}</span>
                    </button>
                )}

                {/* Coordinate Input Toggle */}
                <button
                    className="coord-toggle-button"
                    onClick={() => setShowCoordinateInput(!showCoordinateInput)}
                    title="Enter coordinates directly"
                >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                    </svg>
                </button>
            </div>

            {/* Coordinate Input Panel */}
            {showCoordinateInput && (
                <div className="coordinate-input-panel">
                    <div className="coord-panel-header">
                        <h4>Go to Coordinates</h4>
                        <button className="btn-close-small" onClick={() => setShowCoordinateInput(false)}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M18 6L6 18M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                    <div className="coord-inputs">
                        <div className="coord-input-group">
                            <label>Latitude</label>
                            <input
                                type="text"
                                placeholder="e.g. 39.8 or 39.8N"
                                value={coordLat}
                                onChange={(e) => setCoordLat(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && handleFlyToCoordinates()}
                            />
                        </div>
                        <div className="coord-input-group">
                            <label>Longitude</label>
                            <input
                                type="text"
                                placeholder="e.g. -121.4 or 121.4W"
                                value={coordLon}
                                onChange={(e) => setCoordLon(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && handleFlyToCoordinates()}
                            />
                        </div>
                    </div>
                    <button className="btn btn-primary coord-fly-btn" onClick={handleFlyToCoordinates}>
                        Fly to Location
                    </button>
                    <div className="coord-examples">
                        <p>Quick Test Sites:</p>
                        <button onClick={() => { setCoordLat('40.0'); setCoordLon('-121.4'); }}>Dixie Fire</button>
                        <button onClick={() => { setCoordLat('38.75'); setCoordLon('-120.5'); }}>Caldor Fire</button>
                        <button onClick={() => { setCoordLat('39.75'); setCoordLon('-121.6'); }}>Camp Fire</button>
                    </div>
                </div>
            )}

            {/* Selection mode indicator - shows when selection mode is enabled and not selecting */}
            {selectionModeEnabled && !selectedBounds && !isDrawing && (
                <div className="selection-prompt">
                    <div className="prompt-content">
                        <div className="prompt-icon">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <rect x="3" y="3" width="18" height="18" rx="2" strokeDasharray="4 2" />
                                <path d="M9 3v18M15 3v18M3 9h18M3 15h18" opacity="0.5" />
                            </svg>
                        </div>
                        <h3>Draw Selection</h3>
                        <p>Click and drag on the map to select an area for analysis</p>
                    </div>
                </div>
            )}

            {/* Drawing indicator */}
            {isDrawing && (
                <div className="drawing-indicator">
                    <span>Drawing selection...</span>
                </div>
            )}

            {/* Selection controls - simplified, no export panel */}
            {selectedBounds && !selectionConfirmed && (
                <div className="selection-controls">
                    <h3>Area Selected</h3>
                    <div className="bounds-info">
                        <div className="bounds-row">
                            <span>Area:</span>
                            <span>{calculateArea(selectedBounds)} km²</span>
                        </div>
                        <div className="bounds-row">
                            <span>West:</span>
                            <span>{selectedBounds.west.toFixed(4)}°</span>
                        </div>
                        <div className="bounds-row">
                            <span>South:</span>
                            <span>{selectedBounds.south.toFixed(4)}°</span>
                        </div>
                        <div className="bounds-row">
                            <span>East:</span>
                            <span>{selectedBounds.east.toFixed(4)}°</span>
                        </div>
                        <div className="bounds-row">
                            <span>North:</span>
                            <span>{selectedBounds.north.toFixed(4)}°</span>
                        </div>
                    </div>
                    <div className="selection-actions">
                        <button className="btn" onClick={handleClearSelection}>
                            Redraw
                        </button>
                        <button className="btn btn-primary" onClick={handleConfirmSelection}>
                            Analyze Area
                        </button>
                    </div>
                </div>
            )}

            {/* Analysis Results Panel */}
            {selectionConfirmed && (
                <div className="analysis-panel">
                    <div className="analysis-content">
                        {/* Header */}
                        <div className="analysis-header">
                            <h3>
                                {isAnalyzing ? 'Analyzing...' : analysisResults ? 'Analysis Complete' : 'Analysis'}
                            </h3>
                            <button className="btn-close" onClick={handleClearSelection}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M18 6L6 18M6 6l12 12" />
                                </svg>
                            </button>
                        </div>

                        {/* Loading State */}
                        {isAnalyzing && (
                            <div className="analysis-loading">
                                <div className="loading-spinner" />
                                <p>Downloading satellite imagery...</p>
                                <p className="loading-substep">Running burn severity model...</p>
                                <p className="loading-substep">Generating AI analysis...</p>
                            </div>
                        )}

                        {/* Error State */}
                        {analysisError && (
                            <div className="analysis-error">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="12" cy="12" r="10" />
                                    <path d="M12 8v4M12 16h.01" />
                                </svg>
                                <p>{analysisError}</p>
                                <button className="btn" onClick={handleConfirmSelection}>
                                    Retry Analysis
                                </button>
                            </div>
                        )}

                        {/* Results */}
                        {analysisResults && (
                            <div className="analysis-results">
                                {/* Satellite Image (False Color - Training Config) */}
                                {analysisResults.satellite_image && (
                                    <div className="result-section">
                                        <h4>Satellite (False Color)</h4>
                                        <div className="severity-map-container">
                                            <img
                                                src={analysisResults.satellite_image}
                                                alt="Satellite Image (False Color B5,B4,B3)"
                                                className="severity-map-image"
                                            />
                                            <div className="severity-legend">
                                                <div className="legend-item">
                                                    <span>Sentinel-2 RGB composite (256×256 @ 10m/px)</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Raw Model Output */}
                                {analysisResults.raw_severity_image && (
                                    <div className="result-section">
                                        <h4>Raw Model Output</h4>
                                        <div className="severity-map-container">
                                            <img
                                                src={analysisResults.raw_severity_image}
                                                alt="Raw Model Output (Grayscale)"
                                                className="severity-map-image"
                                            />
                                            <div className="severity-legend">
                                                <div className="legend-item">
                                                    <span className="legend-color-gradient" />
                                                    <span>Black (0%) → White (100%) Burn Severity</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Colorized Severity Map */}
                                <div className="result-section">
                                    <h4>Colorized Severity Map</h4>
                                    <div className="severity-map-container">
                                        <img
                                            src={analysisResults.severity_image}
                                            alt="Burn Severity Map"
                                            className="severity-map-image"
                                        />
                                        <div className="severity-legend">
                                            <div className="legend-item">
                                                <span className="legend-color low" />
                                                <span>None/Low</span>
                                            </div>
                                            <div className="legend-item">
                                                <span className="legend-color moderate" />
                                                <span>Moderate</span>
                                            </div>
                                            <div className="legend-item">
                                                <span className="legend-color high" />
                                                <span>High/Severe</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Statistics */}
                                {analysisResults.severity_stats && (
                                    <div className="result-section">
                                        <h4>Severity Statistics</h4>
                                        <div className="stats-grid">
                                            <div className="stat-item">
                                                <span className="stat-value">
                                                    {(analysisResults.severity_stats.mean_severity * 100).toFixed(1)}%
                                                </span>
                                                <span className="stat-label">Mean Severity</span>
                                            </div>
                                            <div className="stat-item">
                                                <span className="stat-value">
                                                    {(analysisResults.severity_stats.high_severity_ratio * 100).toFixed(1)}%
                                                </span>
                                                <span className="stat-label">High Severity</span>
                                            </div>
                                            <div className="stat-item">
                                                <span className="stat-value">
                                                    {(analysisResults.severity_stats.moderate_severity_ratio * 100).toFixed(1)}%
                                                </span>
                                                <span className="stat-label">Moderate</span>
                                            </div>
                                            <div className="stat-item">
                                                <span className="stat-value">
                                                    {(analysisResults.severity_stats.low_severity_ratio * 100).toFixed(1)}%
                                                </span>
                                                <span className="stat-label">Low/None</span>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Gemini Analysis */}
                                {analysisResults.gemini_analysis && (
                                    <div className="result-section gemini-section">
                                        <h4>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z" />
                                                <path d="M12 16v-4M12 8h.01" />
                                            </svg>
                                            AI Restoration Analysis
                                        </h4>
                                        <div className="gemini-text">
                                            {analysisResults.gemini_analysis.split('\n').map((line, i) => (
                                                <p key={i}>{line}</p>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Actions */}
                                <div className="result-actions">
                                    <button className="btn" onClick={handleClearSelection}>
                                        Analyze New Area
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
