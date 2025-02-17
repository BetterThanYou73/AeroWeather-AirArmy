
var map = L.map('map').setView([53.545, -113.49], 12); // Centered on a reasonable point with zoom level 10

// Add OpenStreetMap tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '© OpenStreetMap'
}).addTo(map);

// Draw a polyline connecting all points (without markers)
var polyline = L.polyline(locations.map(location => location.coords), {
    color: 'blue',   // Line color
    weight: 4,       // Line thickness
    opacity: 0.5     // Transparency
}).addTo(map);

// Add arrows using Leaflet-PolylineDecorator
L.polylineDecorator(polyline, {
    patterns: [
        {
            offset: '10%', // Start a bit into the line
            repeat: '20px', // Distance between arrows
            symbol: L.Symbol.arrowHead({
                pixelSize: 10, // Arrow size
                polygon: false, // Keep arrow as a line
                pathOptions: { stroke: true, color: 'red', weight: 3 }
            })
        }
    ]
}).addTo(map);

// Zoom the map to fit all points
map.fitBounds(polyline.getBounds());
