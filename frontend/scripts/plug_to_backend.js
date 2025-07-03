// plug_to_backend.js

const API_URL = 'http://127.0.0.1:8000/'; // Change to your actual endpoint
const POLL_INTERVAL_MS = 3000; // 5 seconds


async function postDatabaseUpdate() {
    let ENDPOINT = API_URL + 'update_db/run';

    const btn = document.getElementById('launchUpdateButton');
    // Disable and grey out the button
    btn.disabled = true

    const res = await fetch(ENDPOINT, { method: 'POST' });

    if (!res.ok) {
        console.error('Error starting job:', res.statusText);
    } else {
        console.log('Job started successfully');
    }
}

async function abortDatabaseUpdate() {
    let ENDPOINT = API_URL + 'update_db/abort';

    const btn = document.getElementById('abortUpdateButton');
    // Disable and grey out the button
    btn.disabled = true;

    const res = await fetch(ENDPOINT, { method: 'POST' });

    if (!res.ok) {
        console.error('Error aborting job:', res.statusText);
    } else {
        console.log('Job commanded successfully to abort');
    }
}

async function fetchUpdateDatabaseStatus() {
    console.log('Fetching database update status...');

    // Define the proper endpoint URL
    let ENDPOINT = API_URL + 'update_db/status';

    let statusField, lengthField, namesField;
    // Ensure the fields are defined before accessing them
    statusField = document.getElementById('updateStatus');
    lengthField = document.getElementById('updateLength');
    namesField = document.getElementById('updateNames');

    try {
        const response = await fetch(ENDPOINT);
        if (!response.ok) {
            statusField.textContent = `Error: ${response.status}`;
            lengthField.textContent = 'Failed to fetch data';

            throw new Error(`HTTP error upon fetching status: ${response.status}`);
        } else {
            const data = await response.json();
            console.log('Database status:', data);

            // Update the fields with the fetched data
            statusField.textContent = data.status;
            lengthField.textContent = data.updated_conversations.length;
            namesField.innerHTML = data.updated_conversations.length
                ? data.updated_conversations.join('<br>')
                : 'No conversations updated';
        }
    } catch (error) {
        console.error('Error fetching database status:', error.message);
        // Update the fields with error messages
        statusField.textContent = `error`;
        lengthField.textContent = 'error';
        namesField.textContent = `Error: ${error.message}`;
    }

    if (statusField.textContent === 'Running') {
        // If the job is running, disable the button and change its color
        const btn = document.getElementById('launchUpdateButton');
        btn.disabled = true;
        const loadingIcon = document.getElementById('loadingIcon');
        loadingIcon.style.visibility = 'visible';
    } else if (statusField.textContent === 'Aborted') {
        // If the job is aborted, re-enable the abort button
        const abortBtn = document.getElementById('abortUpdateButton');
        abortBtn.disabled = false;
        const runBtn = document.getElementById('launchUpdateButton');
        runBtn.disabled = false;
        const loadingIcon = document.getElementById('loadingIcon');
        loadingIcon.style.visibility = 'hidden';
    } else {
        // If the job is not running, re-enable the button and reset its color
        const btn = document.getElementById('launchUpdateButton');
        btn.disabled = false;
        const loadingIcon = document.getElementById('loadingIcon');
        loadingIcon.style.visibility = 'hidden';
    }
}

// Start periodic polling when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    fetchUpdateDatabaseStatus(); // Optionally fetch once immediately
    setInterval(fetchUpdateDatabaseStatus, POLL_INTERVAL_MS);
});
