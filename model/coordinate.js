fetch("https://myuser.pythonanywhere.com/process", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      elements: window.inputElements // Use inputs from your page
    })
  })
    .then(response => response.json())
    .then(data => {
      console.log("Received data from Flask:", data);
  
      // Now that data is received, loop over the locations array.
      // Make sure the loop is inside the .then() callback.
      const locations = data;
      locations.forEach(([lat, lon]) => {
        console.log(`Latitude: ${lat}, Longitude: ${lon}`);
      });
    })
    .catch(error => console.error("Error:", error));
}