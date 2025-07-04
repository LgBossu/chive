// plug_to_backend.js

const API_URL = 'http://127.0.0.1:8000/'; // Change to your actual endpoint
const POLL_INTERVAL_MS = 3000; // 3 seconds


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
        const loadingIcon = document.getElementById('updateLoadingIcon');
        loadingIcon.src = 'assets/icons/loading_alpha.gif';
    } else if (statusField.textContent === 'Aborted') {
        // If the job is aborted, re-enable the abort button
        const abortBtn = document.getElementById('abortUpdateButton');
        abortBtn.disabled = false;
        const runBtn = document.getElementById('launchUpdateButton');
        runBtn.disabled = false;
        const loadingIcon = document.getElementById('updateLoadingIcon');
        loadingIcon.src = 'assets/icons/applogo.png';
    } else {
        // If the job is not running, re-enable the button and reset its color
        const btn = document.getElementById('launchUpdateButton');
        btn.disabled = false;
        const loadingIcon = document.getElementById('updateLoadingIcon');
        loadingIcon.src = 'assets/icons/applogo.png';
    }
}

async function postCategorization() {
    let ENDPOINT = API_URL + 'categorizer/run';
    const btn = document.getElementById('launchCategorizationButton');
    // Disable and grey out the button
    btn.disabled = true;
    const res = await fetch(ENDPOINT, { method: 'POST' });
    if (!res.ok) {
        console.error('Error starting categorization job:', res.statusText);
    } else {
        console.log('Categorization job started successfully');
    }
}

async function abortCategorization() {
    let ENDPOINT = API_URL + 'categorizer/abort';
    const btn = document.getElementById('abortCategorizationButton');
    // Disable and grey out the button
    btn.disabled = true;
    const res = await fetch(ENDPOINT, { method: 'POST' });
    if (!res.ok) {
        console.error('Error aborting categorization job:', res.statusText);
    } else {
        console.log('Categorization job commanded successfully to abort');
    }
}

async function fetchCategorizationStatus() {
    console.log('Fetching categorization status...');

    // Define the proper endpoint URL
    let ENDPOINT = API_URL + 'categorizer/status';

    let statusField, totalMessagesField, processedMessagesField, etaField, lastUpdateField, currentSpeedField,
        infoField;
    // Ensure the fields are defined before accessing them
    statusField = document.getElementById('categorizationStatus');
    totalMessagesField = document.getElementById('totalMessages');
    processedMessagesField = document.getElementById('processedMessages');
    etaField = document.getElementById('eta');
    lastUpdateField = document.getElementById('lastUpdate');
    currentSpeedField = document.getElementById('currentSpeed');
    infoField = document.getElementById('categorizationInfo');

    try {
        const response = await fetch(ENDPOINT);
        if (!response.ok) {
            statusField.textContent = `Error: ${response.status}`;
            totalMessagesField.textContent = 'Failed to fetch data';
            processedMessagesField.textContent = 'Failed to fetch data';
            etaField.textContent = 'Failed to fetch data';
            lastUpdateField.textContent = 'Failed to fetch data';
            currentSpeedField.textContent = 'Failed to fetch data';
            infoField.textContent = 'Failed to fetch data';

            throw new Error(`HTTP error upon fetching status: ${response.status}`);
        } else {
            const data = await response.json();
            console.log('Categorization status:', data);
            // Update the fields with the fetched data
            statusField.textContent = data.status;
            totalMessagesField.textContent = data.total_messages || 'N/A';
            if (typeof data.processed_messages === 'number' && typeof data.total_messages === 'number' && data.total_messages > 0) {
                const percent = ((data.processed_messages / data.total_messages) * 100).toFixed(2);
                processedMessagesField.textContent = `${data.processed_messages} (${percent}%)`;
            } else {
                processedMessagesField.textContent = data.processed_messages || 'N/A';
            }
            etaField.textContent = data.eta || 'N/A';
            // Convert unix timestamp (seconds) to readable date if present
            if (data.last_update) {
                const date = new Date(data.last_update * 1000);
                lastUpdateField.textContent = date.toLocaleTimeString();
            } else {
                lastUpdateField.textContent = 'N/A';
            }
            currentSpeedField.textContent =
                (typeof data.current_speed === 'number')
                    ? data.current_speed.toFixed(2)
                    : (data.current_speed || 'N/A');
            infoField.innerHTML = 'Nothing to see here !';
        }
    } catch (error) {
        console.error('Error fetching categorization status:', error.message);
        // Update the fields with error messages
        statusField.textContent = `error`;
        totalMessagesField.textContent = 'error';
        processedMessagesField.textContent = 'error';
        etaField.textContent = 'error';
        lastUpdateField.textContent = 'error';
        currentSpeedField.textContent = 'error';
        infoField.textContent = `Error: ${error.message}`;
    }
    if (statusField.textContent === 'Running' || statusField.textContent === 'Stalled') {
        // If the job is running, disable the button and change its color
        const btn = document.getElementById('launchCategorizationButton');
        btn.disabled = true;
        const loadingIcon = document.getElementById('categorizerLoadingIcon');
        loadingIcon.src = 'assets/icons/loading_alpha.gif';
    } else if (statusField.textContent === 'Aborted') {
        // If the job is aborted, re-enable the abort button
        const abortBtn = document.getElementById('abortCategorizationButton');
        abortBtn.disabled = false;
        const runBtn = document.getElementById('launchCategorizationButton');
        runBtn.disabled = false;
        const loadingIcon = document.getElementById('categorizerLoadingIcon');
        loadingIcon.src = 'assets/icons/applogo.png';
    } else {
        // If the job is not running, re-enable the button and reset its color
        const btn = document.getElementById('launchCategorizationButton');
        btn.disabled = false;
        const loadingIcon = document.getElementById('categorizerLoadingIcon');
        loadingIcon.src = 'assets/icons/applogo.png';
    }
}

// Start periodic polling when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    fetchUpdateDatabaseStatus(); // Optionally fetch once immediately
    setInterval(fetchUpdateDatabaseStatus, POLL_INTERVAL_MS);
});

document.addEventListener('DOMContentLoaded', () => {
    fetchCategorizationStatus(); // Optionally fetch once immediately
    setInterval(fetchCategorizationStatus, POLL_INTERVAL_MS);
});
