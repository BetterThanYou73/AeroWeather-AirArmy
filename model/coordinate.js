var Coordinat_arr = [];

fetch("http://127.0.0.1:5000/process", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        elements: window.inputElements  // Use inputs from initial input file
    })
})
.then(response => response.json())  // Convert response to JSON
.then(data => {
    Coordinat_arr = data.Coordinat_arr;  // Assuming response contains a "Coordinat_arr" array
    for (let i = 0; i < Coordinat_arr.length; i++) {
        let [lat, lon] = Coordinat_arr[i]; 
        console.log(`Latitude: ${lat}, Longitude: ${lon}`);
    }
})
.catch(error => console.error("Error:", error));
