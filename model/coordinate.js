
var locations = []
fetch("http://127.0.0.1:5000/process", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        elements: window.inputElements  // Use inputs from initial input file
    })
})
.then(response => response.json())
.then(data => {
    console.log("Received data from Flask:", data);

    // Directly assign API response to locations
    locations = data;  })
for (let i = 0; i < locations.length; i++) {
    let [lat, lon] = locations[i]; 
    console.log(`Latitude: ${lat}, Longitude: ${lon}`);
}