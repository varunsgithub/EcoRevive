import { useState, useEffect, useRef, useCallback } from 'react'
import './InteractiveGlobe.css'
import ErrorBoundary from '../common/ErrorBoundary'

// API Base URL - use environment variable or fallback to localhost
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Simple markdown renderer for chat messages
const renderMarkdown = (text) => {
    if (!text) return ''

    // Check for markdown tables and render them
    const lines = text.split('\n')
    let inTable = false
    let tableRows = []
    let processedLines = []

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim()

        // Detect table rows (starts and ends with |)
        if (line.startsWith('|') && line.endsWith('|')) {
            // Skip separator rows (|---|---|)
            if (line.match(/^\|[\s:-]+\|$/)) {
                continue
            }
            inTable = true
            const cells = line.split('|').filter(c => c.trim()).map(c => c.trim())
            tableRows.push(cells)
        } else {
            if (inTable && tableRows.length > 0) {
                // Render the table
                let tableHtml = '<table class="chat-table"><thead><tr>'
                if (tableRows.length > 0) {
                    tableRows[0].forEach(cell => {
                        tableHtml += `<th>${cell}</th>`
                    })
                    tableHtml += '</tr></thead><tbody>'
                    for (let j = 1; j < tableRows.length; j++) {
                        tableHtml += '<tr>'
                        tableRows[j].forEach(cell => {
                            tableHtml += `<td>${cell}</td>`
                        })
                        tableHtml += '</tr>'
                    }
                    tableHtml += '</tbody></table>'
                }
                processedLines.push(tableHtml)
                tableRows = []
            }
            inTable = false
            processedLines.push(line)
        }
    }

    // Handle table at end of text
    if (tableRows.length > 0) {
        let tableHtml = '<table class="chat-table"><thead><tr>'
        tableRows[0].forEach(cell => {
            tableHtml += `<th>${cell}</th>`
        })
        tableHtml += '</tr></thead><tbody>'
        for (let j = 1; j < tableRows.length; j++) {
            tableHtml += '<tr>'
            tableRows[j].forEach(cell => {
                tableHtml += `<td>${cell}</td>`
            })
            tableHtml += '</tr>'
        }
        tableHtml += '</tbody></table>'
        processedLines.push(tableHtml)
    }

    let html = processedLines.join('\n')
        // Escape HTML (but not our table tags)
        .replace(/&(?!amp;|lt;|gt;)/g, '&amp;')
        .replace(/<(?!\/?(?:table|thead|tbody|tr|th|td)[^>]*>)/g, '&lt;')
        .replace(/>(?!<)/g, function (match, offset, string) {
            // Don't escape > that are part of our table tags
            const before = string.substring(Math.max(0, offset - 10), offset)
            if (before.match(/<\/?(?:table|thead|tbody|tr|th|td)/)) return match
            return '&gt;'
        })
        // Headers (order matters - longest first)
        .replace(/^#### (.+)$/gm, '<h5>$1</h5>')
        .replace(/^### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^## (.+)$/gm, '<h3>$1</h3>')
        .replace(/^# (.+)$/gm, '<h2>$1</h2>')
        // Bold and italic
        .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Bullet points
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/^• (.+)$/gm, '<li>$1</li>')
        .replace(/^\* (.+)$/gm, '<li>$1</li>')
        // Numbered lists
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        // Line breaks
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br/>')

    // Wrap consecutive <li> items in <ul>
    html = html.replace(/(<li>.*?<\/li>)(<br\/>)?/g, '$1')
    html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>')

    return `<p>${html}</p>`
}

// Cesium imports - will be loaded dynamically
let Viewer, Cartesian3, Cartographic, CesiumMath, Rectangle, Color, ScreenSpaceEventHandler, ScreenSpaceEventType, CallbackProperty

/**
 * Interactive Globe Component
 * CesiumJS-powered satellite imagery explorer with drag-to-draw area selection
 * Features: Hope Visualizer, Chat AI, Professional/Personal outputs
 */
function InteractiveGlobeContent({ onBack, isTransitioning, userType = 'personal' }) {
    const cesiumContainerRef = useRef(null)
    const viewerRef = useRef(null)
    const handlerRef = useRef(null)
    const [isLoading, setIsLoading] = useState(true)
    const [cesiumLoaded, setCesiumLoaded] = useState(false)
    const [searchQuery, setSearchQuery] = useState('')
    const [currentAltitude, setCurrentAltitude] = useState(0)
    const [isZoomedIn, setIsZoomedIn] = useState(false)
    const [selectionModeEnabled, setSelectionModeEnabled] = useState(false)
    const [selectedBounds, setSelectedBounds] = useState(null)
    const [isDrawing, setIsDrawing] = useState(false)
    const [drawStart, setDrawStart] = useState(null)
    const [liveArea, setLiveArea] = useState(0)
    const [searchResults, setSearchResults] = useState([])
    const [showSearchResults, setShowSearchResults] = useState(false)
    const [isSearching, setIsSearching] = useState(false)
    const [selectionConfirmed, setSelectionConfirmed] = useState(false)

    // Analysis state
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [analysisResults, setAnalysisResults] = useState(null)
    const [analysisError, setAnalysisError] = useState(null)
    const [analysisStep, setAnalysisStep] = useState(0)

    // UI state for results panel
    const [activeTab, setActiveTab] = useState('overview')

    // Chat state
    const [chatMessages, setChatMessages] = useState([])
    const [chatInput, setChatInput] = useState('')
    const [isChatLoading, setIsChatLoading] = useState(false)
    const [showChat, setShowChat] = useState(false)

    // Export state
    const [isExporting, setIsExporting] = useState(false)
    const [isExportingWord, setIsExportingWord] = useState(false)

    // Coordinate input state
    const [showCoordinateInput, setShowCoordinateInput] = useState(false)
    const [coordLat, setCoordLat] = useState('')
    const [coordLon, setCoordLon] = useState('')

    // Thresholds
    const SELECTION_THRESHOLD = 50000

    // Analysis steps for progress indicator
    const analysisSteps = [
        { id: 1, label: 'Fetching Satellite Data', icon: '01' },
        { id: 2, label: 'Processing Imagery', icon: '02' },
        { id: 3, label: 'Running AI Model', icon: '03' },
        { id: 4, label: 'Gemini Analysis', icon: '04' },
        { id: 5, label: 'Generating Report', icon: '05' }
    ]

    // Quick action prompts based on user type
    const quickActions = userType === 'professional' ? [
        { id: 'legal', label: 'Legal & Tenure', icon: '', prompt: 'Analyze legal and land tenure considerations for this restoration site.' },
        { id: 'biophysical', label: 'Site Analysis', icon: '', prompt: 'Provide detailed biophysical site characterization including soil, hydrology, and slope.' },
        { id: 'species', label: 'Species Palette', icon: '', prompt: 'Recommend native species palette with pioneer vs climax breakdown and drought tolerance.' },
        { id: 'monitoring', label: 'Monitoring Plan', icon: '', prompt: 'Create a monitoring and verification framework with NDVI baselines and carbon accounting.' }
    ] : [
        { id: 'safety', label: 'Safety Check', icon: '', prompt: 'Identify all safety hazards including widowmaker trees, unstable slopes, and danger zones.' },
        { id: 'ownership', label: 'Land Ownership', icon: '', prompt: 'Provide information about land ownership and who to contact for restoration permits.' },
        { id: 'supplies', label: 'Supplies & Cost', icon: '', prompt: 'Generate a list of supplies and estimated costs for a community restoration drive.' }
    ]

    // Debug: Log props on mount
    useEffect(() => {
        console.log('[Globe] Component mounted. onBack type:', typeof onBack, 'userType:', userType)
    }, [])

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

                cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYWE1OWUxNy1mMWZiLTQzYjYtYTQ0OS1kMWFjYmFkNjc5YzciLCJpZCI6NTc3MzMsImlhdCI6MTYyNzI0OTI2NX0.XcKpgANiY19MC4bdFUXMVEBToBmqS8kuYpUlxJHYZxk'

                setCesiumLoaded(true)
            } catch (error) {
                console.error('Failed to load Cesium:', error)
            }
        }
        loadCesium()
    }, [])

    // Initialize Cesium viewer with improved zoom controls
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

                // IMPROVED: Configure zoom settings for Windows trackpad
                const controller = viewer.scene.screenSpaceCameraController
                controller.zoomEventTypes = [
                    cesium.CameraEventType.WHEEL,
                    cesium.CameraEventType.PINCH,
                    cesium.CameraEventType.RIGHT_DRAG
                ]
                // Make zoom more responsive for trackpad
                controller.minimumZoomDistance = 100
                controller.maximumZoomDistance = 50000000
                // Increase zoom sensitivity for trackpad
                controller._zoomFactor = 5.0

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

                    if (altitude < SELECTION_THRESHOLD) {
                        setIsZoomedIn(true)
                    } else {
                        setIsZoomedIn(false)
                        if (!selectedBounds) {
                            setSelectionModeEnabled(false)
                        }
                    }
                })

                // Add custom wheel handler for better trackpad support
                cesiumContainerRef.current.addEventListener('wheel', (e) => {
                    e.preventDefault()
                    const zoomAmount = e.deltaY * -0.001
                    const camera = viewer.camera
                    const height = camera.positionCartographic.height
                    const newHeight = height * (1 - zoomAmount * 2)

                    if (newHeight > 100 && newHeight < 50000000) {
                        camera.zoomIn(height - newHeight)
                    }
                }, { passive: false })

            } catch (error) {
                console.error('Failed to initialize Cesium viewer:', error)
                setIsLoading(false)
            }
        }

        initViewer()

        return () => {
            if (handlerRef.current && !handlerRef.current.isDestroyed()) {
                handlerRef.current.destroy()
            }
            if (viewerRef.current && !viewerRef.current.isDestroyed()) {
                try {
                    viewerRef.current.destroy()
                } catch (e) {
                    console.warn('[Globe] Error destroying viewer:', e)
                }
                viewerRef.current = null
            }
        }
    }, [cesiumLoaded])

    // Disable camera controls when in selection mode
    useEffect(() => {
        if (!viewerRef.current) return

        const viewer = viewerRef.current
        const controller = viewer.scene.screenSpaceCameraController

        if (selectionModeEnabled) {
            controller.enableRotate = false
            controller.enableTranslate = false
            controller.enableTilt = false
            controller.enableZoom = true
        } else {
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

        handler.setInputAction((event) => {
            const cartesian = viewer.camera.pickEllipsoid(event.position, viewer.scene.globe.ellipsoid)
            if (cartesian) {
                const cartographic = Cartographic.fromCartesian(cartesian)
                startPosition = {
                    lon: CesiumMath.toDegrees(cartographic.longitude),
                    lat: CesiumMath.toDegrees(cartographic.latitude)
                }

                currentRectangle = Rectangle.fromDegrees(
                    startPosition.lon,
                    startPosition.lat,
                    startPosition.lon,
                    startPosition.lat
                )

                setDrawStart(startPosition)
                setIsDrawing(true)

                viewer.entities.removeAll()
                setSelectedBounds(null)
                setSelectionConfirmed(false)

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

        handler.setInputAction((event) => {
            if (!startPosition || !currentRectangle) return

            const cartesian = viewer.camera.pickEllipsoid(event.endPosition, viewer.scene.globe.ellipsoid)
            if (cartesian) {
                const cartographic = Cartographic.fromCartesian(cartesian)
                const endLon = CesiumMath.toDegrees(cartographic.longitude)
                const endLat = CesiumMath.toDegrees(cartographic.latitude)

                currentRectangle = Rectangle.fromDegrees(
                    Math.min(startPosition.lon, endLon),
                    Math.min(startPosition.lat, endLat),
                    Math.max(startPosition.lon, endLon),
                    Math.max(startPosition.lat, endLat)
                )

                const west = Math.min(startPosition.lon, endLon)
                const east = Math.max(startPosition.lon, endLon)
                const south = Math.min(startPosition.lat, endLat)
                const north = Math.max(startPosition.lat, endLat)
                const latMid = (north + south) / 2
                const kmPerDegLon = 111.32 * Math.cos(latMid * Math.PI / 180)
                const kmPerDegLat = 110.574
                const width = Math.abs(east - west) * kmPerDegLon
                const height = Math.abs(north - south) * kmPerDegLat
                setLiveArea(width * height)
            }
        }, ScreenSpaceEventType.MOUSE_MOVE)

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
                    viewer.entities.removeAll()
                }
            }
            startPosition = null
            currentRectangle = null
            rectangleEntity = null
            setIsDrawing(false)
            setDrawStart(null)
            setLiveArea(0)
        }, ScreenSpaceEventType.LEFT_UP)

        return () => {
            handler.destroy()
            handlerRef.current = null
        }
    }, [selectionModeEnabled, viewerRef.current])

    // Debounced search effect - auto-search as user types
    useEffect(() => {
        if (!searchQuery.trim()) {
            setSearchResults([])
            setShowSearchResults(false)
            return
        }

        setIsSearching(true)
        const timer = setTimeout(async () => {
            try {
                console.log('[Search] Querying:', searchQuery)
                const response = await fetch(
                    `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=5&addressdetails=1`,
                    {
                        headers: {
                            'Accept': 'application/json'
                        }
                    }
                )
                const results = await response.json()
                console.log('[Search] Results:', results.length, 'items', results)
                setSearchResults(results)
                setShowSearchResults(true)
            } catch (error) {
                console.error('[Search] Failed:', error)
            } finally {
                setIsSearching(false)
            }
        }, 300)

        return () => clearTimeout(timer)
    }, [searchQuery])

    // Manual search on button click
    const handleSearch = useCallback(async () => {
        if (!searchQuery.trim() || isSearching) return

        setIsSearching(true)
        try {
            console.log('[Search Manual] Querying:', searchQuery)
            const response = await fetch(
                `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=5&addressdetails=1`,
                {
                    headers: {
                        'Accept': 'application/json'
                    }
                }
            )
            const results = await response.json()
            console.log('[Search Manual] Results:', results.length, 'items')
            setSearchResults(results)
            setShowSearchResults(true)
        } catch (error) {
            console.error('[Search Manual] Failed:', error)
        } finally {
            setIsSearching(false)
        }
    }, [searchQuery, isSearching])

    // Fly to location
    const flyToLocation = useCallback((lon, lat, name) => {
        console.log('[FlyTo] Location:', name, 'Coords:', lon, lat)
        if (!viewerRef.current) {
            console.error('[FlyTo] No viewer available')
            return
        }

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

    const handleKeyPress = useCallback((e) => {
        if (e.key === 'Enter') {
            handleSearch()
        }
    }, [handleSearch])

    const handleFlyToCoordinates = useCallback(() => {
        if (!viewerRef.current) return

        let lat = parseFloat(coordLat)
        if (coordLat.toLowerCase().includes('s')) {
            lat = -Math.abs(lat)
        }

        let lon = parseFloat(coordLon)
        if (coordLon.toLowerCase().includes('w')) {
            lon = -Math.abs(lon)
        }

        if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
            alert('Invalid coordinates. Latitude must be -90 to 90, Longitude -180 to 180.')
            return
        }

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
        console.log('[Clear] Clearing selection and analysis')
        if (viewerRef.current) {
            viewerRef.current.entities.removeAll()
        }
        setSelectedBounds(null)
        setSelectionConfirmed(false)
        setIsAnalyzing(false)
        setAnalysisResults(null)
        setAnalysisError(null)
        setAnalysisStep(0)
        setActiveTab('overview')
        setChatMessages([])
        setShowChat(false)
    }, [])

    // Auto-clear analysis panel when zooming out
    useEffect(() => {
        if (!isZoomedIn && selectionConfirmed) {
            console.log('[Zoom] Zoomed out - clearing analysis panel')
            handleClearSelection()
        }
    }, [isZoomedIn, selectionConfirmed, handleClearSelection])

    // Confirm selection and trigger analysis
    const handleConfirmSelection = useCallback(async () => {
        if (!selectedBounds) return

        setSelectionConfirmed(true)
        setIsAnalyzing(true)
        setAnalysisError(null)
        setAnalysisResults(null)
        setAnalysisStep(1)

        // Simulate step progression
        const stepInterval = setInterval(() => {
            setAnalysisStep(prev => {
                if (prev >= 5) {
                    clearInterval(stepInterval)
                    return prev
                }
                return prev + 1
            })
        }, 2000)

        try {
            const response = await fetch(`${API_BASE}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    west: selectedBounds.west,
                    south: selectedBounds.south,
                    east: selectedBounds.east,
                    north: selectedBounds.north,
                    user_type: userType
                }),
            })

            const data = await response.json()
            clearInterval(stepInterval)
            setAnalysisStep(5)

            if (data.success) {
                setAnalysisResults(data)

                // Add initial chat message with dynamic stats (prioritize rigorous metrics if available)
                let highSev, meanSev, lowSev
                const stats = data.severity_stats || {}

                if (stats.mean_severity_in_burn_area !== undefined) {
                    highSev = Math.round(stats.high_severity_in_burn_area * 100)
                    meanSev = Math.round(stats.mean_severity_in_burn_area * 100)
                    lowSev = Math.round(stats.low_severity_in_burn_area * 100)
                } else {
                    highSev = Math.round(stats.high_severity_ratio * 100)
                    meanSev = Math.round(stats.mean_severity * 100)
                    lowSev = Math.round(stats.low_severity_ratio * 100)
                }

                const welcomeMsg = userType === 'professional'
                    ? `**Analysis Complete** for ${calculateArea(selectedBounds)} km²\n\n**Severity Breakdown (Within Burn Area):**\n- High severity: ${highSev}%\n- Mean severity: ${meanSev}%\n- Unburned/low: ${lowSev}%\n\nI can help with legal compliance, species recommendations, or monitoring frameworks. Select a quick action or ask me anything.`
                    : `**Analysis Complete** for ${calculateArea(selectedBounds)} km²\n\nI found that ${highSev}% of the burned area has high severity damage, while ${lowSev}% is lightly affected.\n\nWant to know if it's safe to volunteer here? Curious what it could look like in 15 years? Click a button below or just ask me.`

                setChatMessages([{
                    role: 'assistant',
                    content: welcomeMsg
                }])
            } else {
                setAnalysisError(data.error || 'Analysis failed')
            }
        } catch (error) {
            clearInterval(stepInterval)
            setAnalysisError(`Failed to connect to analysis server: ${error.message}`)
        } finally {
            setIsAnalyzing(false)
        }
    }, [selectedBounds, userType])

    // Handle quick action click - DYNAMIC API CALL
    const handleQuickAction = useCallback(async (action) => {
        if (!analysisResults) return

        setIsChatLoading(true)
        setShowChat(true)
        setActiveTab('chat')

        setChatMessages(prev => [...prev, {
            role: 'user',
            content: action.label
        }])

        try {
            // Call the dynamic chat API
            const response = await fetch(`${API_BASE}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: action.prompt,
                    action_type: action.id,
                    user_type: userType,
                    context: {
                        severity_stats: analysisResults.severity_stats,
                        bbox: selectedBounds,
                        layer2_output: analysisResults.layer2_output
                    }
                }),
            })

            const data = await response.json()

            if (data.success && data.response) {
                setChatMessages(prev => [...prev, {
                    role: 'assistant',
                    content: data.response
                }])
            } else {
                throw new Error(data.error || 'Failed to get response')
            }
        } catch (error) {
            console.error('Chat error:', error)
            setChatMessages(prev => [...prev, {
                role: 'assistant',
                content: `Sorry, I encountered an error: ${error.message}. Please make sure the backend server is running.`
            }])
        } finally {
            setIsChatLoading(false)
        }
    }, [analysisResults, selectedBounds, userType])

    // Handle chat submit - DYNAMIC API CALL
    const handleChatSubmit = useCallback(async (e) => {
        e.preventDefault()
        if (!chatInput.trim() || isChatLoading) return

        const userMessage = chatInput.trim()
        setChatInput('')
        setIsChatLoading(true)

        setChatMessages(prev => [...prev, {
            role: 'user',
            content: userMessage
        }])

        try {
            // Call the dynamic chat API
            const response = await fetch(`${API_BASE}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: userMessage,
                    action_type: null, // Free-form question
                    user_type: userType,
                    context: {
                        severity_stats: analysisResults?.severity_stats,
                        bbox: selectedBounds,
                        layer2_output: analysisResults?.layer2_output
                    }
                }),
            })

            const data = await response.json()

            if (data.success && data.response) {
                setChatMessages(prev => [...prev, {
                    role: 'assistant',
                    content: data.response
                }])
            } else {
                throw new Error(data.error || 'Failed to get response')
            }
        } catch (error) {
            console.error('Chat error:', error)
            setChatMessages(prev => [...prev, {
                role: 'assistant',
                content: `Sorry, I encountered an error: ${error.message}. Please make sure the backend server is running.`
            }])
        } finally {
            setIsChatLoading(false)
        }
    }, [chatInput, isChatLoading, analysisResults, selectedBounds, userType])

    // Handle PDF Export
    const handleExportPDF = useCallback(async () => {
        if (!analysisResults || isExporting) return

        setIsExporting(true)

        try {
            const response = await fetch(`${API_BASE}/api/export/pdf`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    satellite_image: analysisResults.satellite_image,
                    severity_image: analysisResults.severity_image,
                    severity_stats: analysisResults.severity_stats,
                    bbox: selectedBounds,
                    layer2_output: analysisResults.layer2_output,
                    layer3_context: analysisResults.layer3_context,
                    carbon_analysis: analysisResults.carbon_analysis,
                    report_type: userType,
                    user_type: userType,
                    location_name: searchQuery || null,
                    analysis_id: null,
                }),
            })

            const data = await response.json()

            if (data.success && data.pdf_base64) {
                // Create download link
                const linkSource = `data:application/pdf;base64,${data.pdf_base64}`
                const downloadLink = document.createElement('a')
                downloadLink.href = linkSource
                downloadLink.download = data.filename || 'EcoRevive_Report.pdf'
                downloadLink.click()
            } else {
                throw new Error(data.error || 'Failed to generate PDF')
            }
        } catch (error) {
            console.error('PDF export error:', error)
            alert(`Failed to export PDF: ${error.message}`)
        } finally {
            setIsExporting(false)
        }
    }, [analysisResults, selectedBounds, userType, searchQuery, isExporting])

    // Handle Word Export
    const handleExportWord = useCallback(async () => {
        if (!analysisResults || isExportingWord) return

        setIsExportingWord(true)

        try {
            const response = await fetch(`${API_BASE}/api/export/word`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    satellite_image: analysisResults.satellite_image,
                    severity_image: analysisResults.severity_image,
                    severity_stats: analysisResults.severity_stats,
                    bbox: selectedBounds,
                    layer2_output: analysisResults.layer2_output,
                    layer3_context: analysisResults.layer3_context,
                    carbon_analysis: analysisResults.carbon_analysis,
                    report_type: userType,
                    user_type: userType,
                    location_name: searchQuery || null,
                }),
            })

            const data = await response.json()

            if (data.success && data.docx_base64) {
                const linkSource = `data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,${data.docx_base64}`
                const downloadLink = document.createElement('a')
                downloadLink.href = linkSource
                downloadLink.download = data.filename || 'EcoRevive_Report.docx'
                downloadLink.click()
            } else {
                throw new Error(data.error || 'Failed to generate report')
            }
        } catch (error) {
            console.error('Export error:', error)
            alert(`Failed to export report: ${error.message}`)
        } finally {
            setIsExportingWord(false)
        }
    }, [analysisResults, selectedBounds, userType, searchQuery, isExportingWord])

    const formatAltitude = (meters) => {
        if (meters > 1000000) {
            return `${(meters / 1000000).toFixed(1)}M km`
        } else if (meters > 1000) {
            return `${(meters / 1000).toFixed(1)} km`
        }
        return `${meters.toFixed(0)} m`
    }

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
                    <p>Initializing Satellite View</p>
                    <span className="loading-hint">Use scroll or pinch to zoom</span>
                </div>
            )}

            {/* Cesium container */}
            <div ref={cesiumContainerRef} className="cesium-container" />

            {/* Top bar */}
            <div className="globe-topbar">
                <button
                    type="button"
                    className="back-button"
                    onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        console.log('[Back] Button clicked')
                        if (typeof onBack === 'function') {
                            onBack()
                        } else {
                            console.error('[Back] onBack is not a function:', onBack)
                        }
                    }}
                >
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
                    <button className="search-button" onClick={handleSearch} disabled={isSearching}>
                        {isSearching ? (
                            <div className="search-spinner" />
                        ) : (
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="11" cy="11" r="8" />
                                <path d="M21 21l-4.35-4.35" />
                            </svg>
                        )}
                    </button>

                    {showSearchResults && searchResults.length > 0 && (
                        <div className="search-results">
                            {searchResults.map((result, index) => (
                                <button
                                    key={index}
                                    type="button"
                                    className="search-result-item"
                                    onClick={(e) => {
                                        e.preventDefault()
                                        e.stopPropagation()
                                        flyToLocation(result.lon, result.lat, result.display_name)
                                    }}
                                >
                                    <span className="result-icon">📍</span>
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

                {/* Zoom hint for trackpad users */}
                <div className="zoom-hint">
                    <span>Scroll to zoom</span>
                </div>

                {isZoomedIn && !selectedBounds && (
                    <button
                        className={`selection-toggle-button ${selectionModeEnabled ? 'active' : ''}`}
                        onClick={() => setSelectionModeEnabled(!selectionModeEnabled)}
                    >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" strokeDasharray={selectionModeEnabled ? "0" : "4 2"} />
                            {selectionModeEnabled && <path d="M9 12l2 2 4-4" />}
                        </svg>
                        <span>{selectionModeEnabled ? 'Drawing' : 'Select Area'}</span>
                    </button>
                )}

                <button
                    className="coord-toggle-button"
                    onClick={() => setShowCoordinateInput(!showCoordinateInput)}
                    title="Enter coordinates or use test sites"
                >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                    </svg>
                </button>
            </div>

            {/* Quick Test Sites Banner */}
            {!selectedBounds && !showCoordinateInput && (
                <div className="test-sites-banner">
                    <span className="banner-label">Try a California Fire Site:</span>
                    <div className="test-site-buttons">
                        <button onClick={() => flyToLocation(-121.4, 40.0, 'Dixie Fire, CA')} className="test-site-btn">
                            Dixie Fire
                        </button>
                        <button onClick={() => flyToLocation(-120.5, 38.75, 'Caldor Fire, CA')} className="test-site-btn">
                            Caldor Fire
                        </button>
                        <button onClick={() => flyToLocation(-121.6, 39.75, 'Camp Fire, CA')} className="test-site-btn">
                            Camp Fire
                        </button>
                    </div>
                </div>
            )}

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
                </div>
            )}

            {/* Selection mode indicator */}
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

            {/* Drawing indicator with live area */}
            {isDrawing && (
                <div className="drawing-indicator">
                    <span className="live-area valid">
                        {liveArea.toFixed(1)} km²
                    </span>
                </div>
            )}

            {/* Selection controls */}
            {selectedBounds && !selectionConfirmed && (
                <div className="selection-controls">
                    <div className="selection-header">
                        <div className="selection-icon"></div>
                        <div>
                            <h3>Area Selected</h3>
                            <span className="area-badge">
                                {calculateArea(selectedBounds)} km²
                            </span>
                        </div>
                    </div>

                    <div className="bounds-info">
                        <div className="bounds-grid">
                            <div className="bound-item">
                                <span className="bound-label">N</span>
                                <span className="bound-value">{selectedBounds.north.toFixed(4)}°</span>
                            </div>
                            <div className="bound-item">
                                <span className="bound-label">S</span>
                                <span className="bound-value">{selectedBounds.south.toFixed(4)}°</span>
                            </div>
                            <div className="bound-item">
                                <span className="bound-label">E</span>
                                <span className="bound-value">{selectedBounds.east.toFixed(4)}°</span>
                            </div>
                            <div className="bound-item">
                                <span className="bound-label">W</span>
                                <span className="bound-value">{selectedBounds.west.toFixed(4)}°</span>
                            </div>
                        </div>
                    </div>

                    <div className="selection-actions">
                        <button className="btn btn-secondary" onClick={handleClearSelection}>
                            Redraw
                        </button>
                        <button
                            className="btn btn-primary btn-analyze"
                            onClick={handleConfirmSelection}
                        >
                            <span>Analyze Area</span>
                        </button>
                    </div>
                </div>
            )}

            {/* Analysis Results Panel - THE MAIN EVENT */}
            {selectionConfirmed && (
                <div className={`analysis-panel ${showChat ? 'with-chat' : ''}`}>
                    {/* Header */}
                    <div className="analysis-header">
                        <div className="header-left">
                            <div>
                                <h3>{isAnalyzing ? 'Analyzing...' : analysisResults ? 'Analysis' : 'Analysis Failed'}</h3>
                                <span className="header-subtitle">{calculateArea(selectedBounds)} km² • {userType === 'professional' ? 'Professional' : 'Personal'}</span>
                            </div>
                        </div>
                        <button type="button" className="btn-close" onClick={handleClearSelection}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M18 6L6 18M6 6l12 12" />
                            </svg>
                        </button>
                    </div>

                    {/* Loading State with Steps */}
                    {isAnalyzing && (
                        <div className="analysis-loading">
                            <div className="loading-steps">
                                {analysisSteps.map((step) => (
                                    <div
                                        key={step.id}
                                        className={`loading-step ${analysisStep >= step.id ? 'active' : ''} ${analysisStep === step.id ? 'current' : ''}`}
                                    >
                                        <span className="step-icon">{step.icon}</span>
                                        <span className="step-label">{step.label}</span>
                                        {analysisStep === step.id && <div className="step-spinner" />}
                                        {analysisStep > step.id && <span className="step-check">Done</span>}
                                    </div>
                                ))}
                            </div>
                            <div className="loading-progress">
                                <div className="progress-bar" style={{ width: `${(analysisStep / 5) * 100}%` }} />
                            </div>
                        </div>
                    )}

                    {/* Error State */}
                    {analysisError && (
                        <div className="analysis-error">
                            <div className="error-icon">X</div>
                            <h4>Analysis Failed</h4>
                            <p>{analysisError}</p>
                            <button className="btn btn-primary" onClick={handleConfirmSelection}>
                                Retry Analysis
                            </button>
                        </div>
                    )}

                    {/* Results Content */}
                    {analysisResults && (
                        <div className="analysis-content">
                            {/* Layer 3 Contextual Warning Banner */}
                            {analysisResults.layer3_context && analysisResults.layer3_context.overall_caution_level !== 'none' && (
                                <div className={`context-warning-banner caution-${analysisResults.layer3_context.overall_caution_level}`}>
                                    <div className="warning-icon">
                                        {'!'}
                                    </div>
                                    <div className="warning-content">
                                        <div className="warning-header">
                                            <span className="warning-badge">
                                                {analysisResults.layer3_context.land_use?.land_use_type?.toUpperCase() || 'UNKNOWN'} AREA
                                            </span>
                                            <span className="caution-level">
                                                {analysisResults.layer3_context.overall_caution_level === 'high' ? 'High Caution' :
                                                    analysisResults.layer3_context.overall_caution_level === 'moderate' ? 'Moderate Caution' : 'Low Caution'}
                                            </span>
                                        </div>
                                        <p className="warning-message">
                                            {analysisResults.layer3_context.land_use?.caution_message || analysisResults.layer3_context.user_guidance}
                                        </p>
                                        {analysisResults.layer3_context.land_use?.recommendations?.length > 0 && (
                                            <ul className="warning-recommendations">
                                                {analysisResults.layer3_context.land_use.recommendations.slice(0, 2).map((rec, i) => (
                                                    <li key={i}>{rec}</li>
                                                ))}
                                            </ul>
                                        )}
                                    </div>
                                    <button
                                        className="warning-dismiss"
                                        onClick={(e) => {
                                            e.target.closest('.context-warning-banner').style.display = 'none'
                                        }}
                                    >
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d="M18 6L6 18M6 6l12 12" />
                                        </svg>
                                    </button>
                                </div>
                            )}

                            {/* Tab Navigation */}
                            <div className="results-tabs">
                                <button
                                    className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('overview')}
                                >
                                    <span></span>
                                    <span>Overview</span>
                                </button>
                                <button
                                    className={`tab-btn ${activeTab === 'imagery' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('imagery')}
                                >
                                    <span></span>
                                    <span>Imagery</span>
                                </button>
                                <button
                                    className={`tab-btn ${activeTab === 'carbon' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('carbon')}
                                >
                                    <span></span>
                                    <span>Carbon</span>
                                </button>
                                <button
                                    className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
                                    onClick={() => { setActiveTab('chat'); setShowChat(true); }}
                                >
                                    <span></span>
                                    <span>AI Chat</span>
                                </button>
                            </div>

                            {/* Tab Content */}
                            <div className="tab-content">
                                {/* Overview Tab */}
                                {activeTab === 'overview' && (
                                    <div className="tab-overview">
                                        {/* Key Stats */}
                                        <div className="stats-hero">
                                            <div className="stat-card primary">
                                                <div className="stat-icon"></div>
                                                <div className="stat-content">
                                                    <span className="stat-value">
                                                        {analysisResults.severity_stats ? Math.round(analysisResults.severity_stats.mean_severity * 100) : 0}%
                                                    </span>
                                                    <span className="stat-label">Mean Burn Severity</span>
                                                </div>
                                            </div>
                                            <div className="stat-card">
                                                <div className="stat-content">
                                                    <span className="stat-value">
                                                        {analysisResults.severity_stats ? Math.round(analysisResults.severity_stats.high_severity_ratio * 100) : 0}%
                                                    </span>
                                                    <span className="stat-label">High Severity</span>
                                                </div>
                                            </div>
                                            <div className="stat-card">
                                                <div className="stat-content">
                                                    <span className="stat-value">
                                                        {analysisResults.severity_stats ? Math.round(analysisResults.severity_stats.moderate_severity_ratio * 100) : 0}%
                                                    </span>
                                                    <span className="stat-label">Moderate</span>
                                                </div>
                                            </div>
                                            <div className="stat-card healthy">
                                                <div className="stat-content">
                                                    <span className="stat-value">
                                                        {analysisResults.severity_stats ? Math.round(analysisResults.severity_stats.low_severity_ratio * 100) : 0}%
                                                    </span>
                                                    <span className="stat-label">Low/Unburned</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Severity Map Preview */}
                                        {analysisResults.severity_image && (
                                            <div className="map-preview">
                                                <h4>Severity Map</h4>
                                                <img src={analysisResults.severity_image} alt="Burn Severity" />
                                                <div className="severity-legend-bar">
                                                    <span>Low</span>
                                                    <div className="legend-gradient" />
                                                    <span>High</span>
                                                </div>
                                            </div>
                                        )}

                                        {/* Land Use Context (Layer 3) */}
                                        {analysisResults.layer3_context?.land_use && (
                                            <div className="land-use-info">
                                                <div className="land-use-header">
                                                    <span className="land-use-icon"></span>
                                                    <div>
                                                        <span className="land-use-type">
                                                            {analysisResults.layer3_context.land_use.land_use_type?.charAt(0).toUpperCase() +
                                                                analysisResults.layer3_context.land_use.land_use_type?.slice(1) || 'Unknown'} Area
                                                        </span>
                                                        <span className={`reliability-badge ${analysisResults.layer3_context.analysis_suitable ? 'good' : 'caution'}`}>
                                                            {analysisResults.layer3_context.analysis_suitable ? 'High Reliability' : 'Use with Caution'}
                                                        </span>
                                                    </div>
                                                </div>
                                                {analysisResults.layer3_context.land_use.land_use_description && (
                                                    <p className="land-use-description">
                                                        {analysisResults.layer3_context.land_use.land_use_description}
                                                    </p>
                                                )}
                                            </div>
                                        )}

                                        {/* AI Summary */}
                                        {analysisResults.gemini_analysis && (
                                            <div className="ai-summary">
                                                <div className="summary-header">
                                                    <span className="gemini-badge">Gemini Analysis</span>
                                                </div>
                                                <div className="summary-content">
                                                    {analysisResults.gemini_analysis.split('\n').slice(0, 5).map((line, i) => (
                                                        line.trim() && <p key={i}>{line}</p>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {/* Quick Actions */}
                                        <div className="quick-actions">
                                            <h4>Quick Actions</h4>
                                            <div className="action-grid">
                                                {quickActions.map(action => (
                                                    <button
                                                        key={action.id}
                                                        className="action-btn"
                                                        onClick={() => handleQuickAction(action)}
                                                    >
                                                        <span className="action-icon">{action.icon}</span>
                                                        <span className="action-label">{action.label}</span>
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Imagery Tab */}
                                {activeTab === 'imagery' && (
                                    <div className="tab-imagery">
                                        <div className="imagery-grid">
                                            {analysisResults.satellite_image && (
                                                <div className="imagery-card">
                                                    <h4>Satellite (False Color)</h4>
                                                    <img src={analysisResults.satellite_image} alt="Satellite" />
                                                    <p>Sentinel-2 RGB composite (256×256 @ 10m/px)</p>
                                                </div>
                                            )}
                                            {analysisResults.raw_severity_image && (
                                                <div className="imagery-card">
                                                    <h4>Raw Model Output</h4>
                                                    <img src={analysisResults.raw_severity_image} alt="Raw Model" />
                                                    <p>U-Net burn severity prediction (0-1)</p>
                                                </div>
                                            )}
                                            {analysisResults.severity_image && (
                                                <div className="imagery-card">
                                                    <h4>Burn Severity Map</h4>
                                                    <img src={analysisResults.severity_image} alt="Severity Map" />
                                                    <p>Heat map: Yellow (low) → Red (high)</p>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}

                                {/* Carbon Calculator Tab */}
                                {activeTab === 'carbon' && analysisResults.carbon_analysis && (
                                    <div className="tab-carbon">
                                        {/* Personal User View */}
                                        {userType === 'personal' ? (
                                            <div className="carbon-personal">
                                                <div className="carbon-hero">
                                                    <div className="carbon-icon-large"></div>
                                                    <div className="carbon-headline">
                                                        <span className="carbon-number">
                                                            {parseInt(analysisResults.carbon_analysis.personal?.total_co2_capture_20yr || 0).toLocaleString()}
                                                        </span>
                                                        <span className="carbon-unit">tons CO₂</span>
                                                    </div>
                                                    <p className="carbon-subheadline">
                                                        could be captured over 20 years of restoration
                                                    </p>
                                                </div>

                                                <div className="carbon-equivalencies">
                                                    <h4>That's equivalent to...</h4>
                                                    <div className="equivalency-grid">
                                                        <div className="equivalency-card">
                                                            <span className="eq-icon"></span>
                                                            <span className="eq-number">
                                                                {Math.round(analysisResults.carbon_analysis.personal?.equivalencies?.cars_off_road_for_year || 0).toLocaleString()}
                                                            </span>
                                                            <span className="eq-label">cars off the road for a year</span>
                                                        </div>
                                                        <div className="equivalency-card">
                                                            <span className="eq-icon"></span>
                                                            <span className="eq-number">
                                                                {(analysisResults.carbon_analysis.personal?.equivalencies?.tree_seedlings_grown_10yr || 0).toLocaleString()}
                                                            </span>
                                                            <span className="eq-label">tree seedlings grown 10 years</span>
                                                        </div>
                                                        <div className="equivalency-card">
                                                            <span className="eq-icon"></span>
                                                            <span className="eq-number">
                                                                {Math.round(analysisResults.carbon_analysis.personal?.equivalencies?.round_trip_flights_nyc_la || 0).toLocaleString()}
                                                            </span>
                                                            <span className="eq-label">NYC↔LA flights offset</span>
                                                        </div>
                                                        <div className="equivalency-card">
                                                            <span className="eq-icon"></span>
                                                            <span className="eq-number">
                                                                {Math.round(analysisResults.carbon_analysis.personal?.equivalencies?.homes_electricity_year || 0).toLocaleString()}
                                                            </span>
                                                            <span className="eq-label">homes' electricity for a year</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="carbon-impact-statements">
                                                    {analysisResults.carbon_analysis.personal?.impact_statements?.map((statement, i) => (
                                                        <p key={i} className="impact-statement">
                                                            {''}
                                                            {statement}
                                                        </p>
                                                    ))}
                                                </div>

                                                <div className="carbon-cta">
                                                    <p>Ready to make an impact?</p>
                                                    <button
                                                        className="btn btn-primary"
                                                        onClick={() => handleQuickAction(quickActions.find(a => a.id === 'supplies') || quickActions[0])}
                                                    >
                                                        Start Restoring
                                                    </button>
                                                </div>
                                            </div>
                                        ) : (
                                            /* Professional User View */
                                            <div className="carbon-professional">
                                                <div className="carbon-pro-header">
                                                    <h4>Carbon Accounting Analysis</h4>
                                                    <span className="methodology-badge">
                                                        {analysisResults.carbon_analysis.professional?.methodology || 'IPCC Tier 2'}
                                                    </span>
                                                </div>

                                                <div className="carbon-metrics-grid">
                                                    <div className="carbon-metric">
                                                        <span className="metric-label">Annual Sequestration</span>
                                                        <span className="metric-value">
                                                            {analysisResults.carbon_analysis.professional?.annual_sequestration_tco2e || 0}
                                                            <span className="metric-unit">tCO₂e/year</span>
                                                        </span>
                                                    </div>
                                                    <div className="carbon-metric">
                                                        <span className="metric-label">Baseline Carbon Stock</span>
                                                        <span className="metric-value">
                                                            {analysisResults.carbon_analysis.professional?.baseline_carbon_tc || 0}
                                                            <span className="metric-unit">tC</span>
                                                        </span>
                                                    </div>
                                                    <div className="carbon-metric">
                                                        <span className="metric-label">Carbon Lost to Fire</span>
                                                        <span className="metric-value warning">
                                                            -{analysisResults.carbon_analysis.professional?.carbon_lost_tc || 0}
                                                            <span className="metric-unit">tC</span>
                                                        </span>
                                                    </div>
                                                    <div className="carbon-metric">
                                                        <span className="metric-label">Current Carbon Stock</span>
                                                        <span className="metric-value">
                                                            {analysisResults.carbon_analysis.professional?.current_carbon_tc || 0}
                                                            <span className="metric-unit">tC</span>
                                                        </span>
                                                    </div>
                                                </div>

                                                <div className="carbon-projections">
                                                    <h5>Sequestration Projections</h5>
                                                    <div className="projections-table">
                                                        <div className="table-header">
                                                            <span>Years</span>
                                                            <span>Cumulative tCO₂e</span>
                                                            <span>Annual Rate</span>
                                                        </div>
                                                        {analysisResults.carbon_analysis.professional?.projections?.map((proj, i) => (
                                                            <div key={i} className="table-row">
                                                                <span>{proj.years} yr</span>
                                                                <span>{proj.cumulative_tco2e?.toLocaleString()}</span>
                                                                <span>{proj.annual_rate_tco2e} tCO₂e/yr</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>

                                                <div className="carbon-protocols">
                                                    <h5>Protocol Eligibility</h5>
                                                    <div className="protocols-grid">
                                                        {Object.entries(analysisResults.carbon_analysis.professional?.protocols || {}).map(([protocol, eligible]) => (
                                                            <div key={protocol} className={`protocol-badge ${eligible ? 'eligible' : 'ineligible'}`}>
                                                                <span className="protocol-status">{eligible ? 'Yes' : 'No'}</span>
                                                                <span className="protocol-name">
                                                                    {protocol.replace(/_/g, ' ').replace('eligible', '').trim()}
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>

                                                <div className="carbon-uncertainty">
                                                    <h5>Uncertainty Quantification</h5>
                                                    <div className="uncertainty-info">
                                                        <div className="uncertainty-bar">
                                                            <div className="uncertainty-range" style={{
                                                                left: '15%',
                                                                width: '70%'
                                                            }}>
                                                                <span className="ci-label">95% CI</span>
                                                            </div>
                                                            <div className="uncertainty-point" style={{ left: '50%' }} />
                                                        </div>
                                                        <div className="uncertainty-values">
                                                            <span>{analysisResults.carbon_analysis.professional?.confidence_interval_low?.toLocaleString()} tCO₂e</span>
                                                            <span className="ci-center">20-year projection</span>
                                                            <span>{analysisResults.carbon_analysis.professional?.confidence_interval_high?.toLocaleString()} tCO₂e</span>
                                                        </div>
                                                        <p className="uncertainty-note">
                                                            Uncertainty: ±{analysisResults.carbon_analysis.professional?.uncertainty_pct}% (combined sources)
                                                        </p>
                                                    </div>
                                                </div>

                                                <div className="carbon-limitations">
                                                    <h5>Limitations & Data Sources</h5>
                                                    <details>
                                                        <summary>View methodology details</summary>
                                                        <div className="limitations-content">
                                                            <p><strong>Limitations:</strong></p>
                                                            <ul>
                                                                {analysisResults.carbon_analysis.professional?.limitations?.map((lim, i) => (
                                                                    <li key={i}>{lim}</li>
                                                                ))}
                                                            </ul>
                                                            <p><strong>Data Sources:</strong></p>
                                                            <ul>
                                                                {analysisResults.carbon_analysis.professional?.data_sources?.map((src, i) => (
                                                                    <li key={i}>{src}</li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    </details>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* Chat Tab */}
                                {activeTab === 'chat' && (
                                    <div className="tab-chat">
                                        <div className="chat-container">
                                            <div className="chat-messages">
                                                {chatMessages.map((msg, i) => (
                                                    <div key={i} className={`chat-message ${msg.role}`}>
                                                        {msg.role === 'assistant' && (
                                                            <div className="message-avatar">AI</div>
                                                        )}
                                                        <div
                                                            className="message-content markdown-content"
                                                            dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                                                        />
                                                    </div>
                                                ))}
                                                {isChatLoading && (
                                                    <div className="chat-message assistant loading">
                                                        <div className="message-avatar">AI</div>
                                                        <div className="message-content">
                                                            <div className="typing-indicator">
                                                                <span></span>
                                                                <span></span>
                                                                <span></span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>

                                            {/* Quick Action Buttons in Chat */}
                                            <div className="chat-quick-actions">
                                                {quickActions.map(action => (
                                                    <button
                                                        key={action.id}
                                                        className="quick-action-chip"
                                                        onClick={() => handleQuickAction(action)}
                                                        disabled={isChatLoading}
                                                    >
                                                        <span>{action.icon}</span>
                                                        <span>{action.label}</span>
                                                    </button>
                                                ))}
                                            </div>

                                            <form className="chat-input-form" onSubmit={handleChatSubmit}>
                                                <input
                                                    type="text"
                                                    value={chatInput}
                                                    onChange={(e) => setChatInput(e.target.value)}
                                                    placeholder="Ask about restoration, species, costs..."
                                                    disabled={isChatLoading}
                                                />
                                                <button type="submit" disabled={isChatLoading || !chatInput.trim()}>
                                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                        <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
                                                    </svg>
                                                </button>
                                            </form>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Footer Actions */}
                            <div className="results-footer">
                                <button className="btn btn-secondary" onClick={handleClearSelection}>
                                    New Analysis
                                </button>
                                <button
                                    className="btn btn-export"
                                    onClick={handleExportWord}
                                    disabled={isExportingWord}
                                >
                                    <span>{isExportingWord ? 'Exporting...' : 'Download Report'}</span>
                                </button>
                                <button className="btn btn-primary" onClick={() => { setActiveTab('chat'); setShowChat(true); }}>
                                    <span>AI Chat</span>
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* User Type Indicator */}
            <div className="user-type-indicator">
                <span className={`user-badge ${userType}`}>
                    {userType === 'professional' ? 'Professional' : 'Personal'}
                </span>
            </div>
        </div>
    )
}

export default function InteractiveGlobe(props) {
    return (
        <ErrorBoundary
            fallback={
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100vh',
                    background: '#050508',
                    color: 'white'
                }}>
                    <h3>Global View Error</h3>
                    <p>The satellite view encountered a problem.</p>
                    <button
                        onClick={props.onBack}
                        style={{
                            marginTop: '16px',
                            padding: '8px 16px',
                            background: '#333',
                            border: '1px solid #555',
                            borderRadius: '4px',
                            color: 'white',
                            cursor: 'pointer'
                        }}
                    >
                        Return to Menu
                    </button>
                </div>
            }
        >
            <InteractiveGlobeContent {...props} />
        </ErrorBoundary>
    )
}
