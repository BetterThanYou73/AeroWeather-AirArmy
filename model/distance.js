const readline = require("readline"); // Import readline module for user input

const rl = readline.createInterface({
    input: process.stdin, // Accept input from the terminal
    output: process.stdout // Display output in the terminal
});

/**
 * Calculates the great-circle distance between two points on Earth using the Haversine formula.
 * 
 * Formula: 
 * d = 2 * R * atan2( sqrt(a), sqrt(1-a) )
 * where:
 * a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
 *
 * @param {number} lat1 - Latitude of the first point in degrees
 * @param {number} lon1 - Longitude of the first point in degrees
 * @param {number} lat2 - Latitude of the second point in degrees
 * @param {number} lon2 - Longitude of the second point in degrees
 * @returns {number} Distance between two points in kilometers
 */
function haversine(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in kilometers
    const toRad = (angle) => angle * (Math.PI / 180); // Convert degrees to radians
    const deltaLat = toRad(lat2 - lat1); // Compute latitude difference in radians
    const deltaLon = toRad(lon2 - lon1); // Compute longitude difference in radians

    const a = Math.sin(deltaLat / 2) ** 2 + 
              Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * 
              Math.sin(deltaLon / 2) ** 2; // Haversine formula part

    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)); // Compute central angle
    return R * c; // Calculate and return distance in km
}

rl.question("Enter your current latitude: ", (lat1) => { // Ask user for current latitude
    rl.question("Enter your current longitude: ", (lon1) => { // Ask user for current longitude
        rl.question("Enter your destination latitude: ", (lat2) => { // Ask user for destination latitude
            rl.question("Enter your destination longitude: ", (lon2) => { // Ask user for destination longitude

                lat1 = parseFloat(lat1); // Convert input to a floating-point number
                lon1 = parseFloat(lon1); // Convert input to a floating-point number
                lat2 = parseFloat(lat2); // Convert input to a floating-point number
                lon2 = parseFloat(lon2); // Convert input to a floating-point number

                if (isNaN(lat1) || isNaN(lon1) || isNaN(lat2) || isNaN(lon2)) { // Check for invalid inputs
                    console.log("Invalid input! Please enter valid numbers."); // Display error if input is not a number
                } else {
                    const distance = haversine(lat1, lon1, lat2, lon2); // Call haversine function
                    console.log(`Distance: ${distance.toFixed(2)} km`); // Print the calculated distance
                }

                rl.close(); // Close the input stream
            });
        });
    });
});
