// plug_to_backend.js

const API_URL = 'http://localhost:8000/'; // Change to your actual endpoint
const POLL_INTERVAL_MS = 2000; // 5 seconds

async function postDatabaseUpdate() {
    try {
        const response = await fetch(API_URL);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Database update response:', data);
    } catch (error) {
        console.error('Error updating database:', error.message);
    }
}

async function fetchUpdateDatabaseStatus() {
    try {
        const response = await fetch(API_URL);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Database status:', data);
    } catch (error) {
        console.error('Error fetching database status:', error.message);
    }
}

// Periodic polling
setInterval(fetchAndDisplay, POLL_INTERVAL_MS);