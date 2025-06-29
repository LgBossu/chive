// plug_to_backend.js

const API_URL = 'http://localhost:3000/api/endpoint'; // Change to your actual endpoint
const POLL_INTERVAL_MS = 5000; // 5 seconds

async function fetchAndDisplay() {
    try {
        const response = await fetch(API_URL);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        console.clear();
        console.log('API Response:', data);
    } catch (error) {
        console.error('Error fetching API:', error.message);
    }
}

// Initial fetch
fetchAndDisplay();
// Periodic polling
setInterval(fetchAndDisplay, POLL_INTERVAL_MS);