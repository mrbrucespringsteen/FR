// Main site JavaScript
// Main site JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('FR site script loaded successfully');
    
    // Test API connection when the page loads
    testApiConnection();
    
    // Add event listeners for any buttons that should trigger API calls
    const generateButton = document.getElementById('generate-button');
    if (generateButton) {
        generateButton.addEventListener('click', handleGenerateClick);
    }
    
    const singleDrawButton = document.getElementById('single-draw-button');
    if (singleDrawButton) {
        singleDrawButton.addEventListener('click', handleSingleDrawClick);
    }
});

// Function to test API connection
function testApiConnection() {
    // Get API endpoint from config.js (which should be included before this script)
    const apiUrl = window.API_ENDPOINT || 'http://api.friendshipresearch.org:5001';
    
    console.log('Testing connection to API at:', apiUrl);
    
    fetch(`${apiUrl}/api/test`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        },
        mode: 'cors'
    })
    .then(response => {
        console.log('API response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('API test successful:', data);
        // Optionally display a success message on the page
        displayApiStatus('API connection successful!', true);
    })
    .catch(error => {
        console.error('API test failed:', error);
        // Optionally display an error message on the page
        displayApiStatus('API connection failed. See console for details.', false);
    });
}

// Function to handle the generate button click
function handleGenerateClick() {
    const input1 = parseFloat(document.getElementById('input1').value) || 0;
    const input2 = parseFloat(document.getElementById('input2').value) || 1;
    
    const apiUrl = window.API_ENDPOINT || 'http://api.friendshipresearch.org:5001';
    
    fetch(`${apiUrl}/api/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input1, input2 }),
        mode: 'cors'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success && data.image) {
            // Display the image
            const resultDiv = document.getElementById('result-display');
            if (resultDiv) {
                resultDiv.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Generated visualization" style="max-width:100%;">`;
            }
        } else {
            console.error('API error:', data.error);
            alert('Error generating visualization. See console for details.');
        }
    })
    .catch(error => {
        console.error('API request failed:', error);
        alert('Failed to communicate with the API.');
    });
}

// Function to handle the single draw button click
function handleSingleDrawClick() {
    const input1 = parseFloat(document.getElementById('input1').value) || 0;
    const input2 = parseFloat(document.getElementById('input2').value) || 1;
    
    const apiUrl = window.API_ENDPOINT || 'http://api.friendshipresearch.org:5001';
    
    fetch(`${apiUrl}/api/single`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input1, input2 }),
        mode: 'cors'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Display the result
            const resultDiv = document.getElementById('result-display');
            if (resultDiv) {
                resultDiv.innerHTML = `<h3>Result: ${data.result}</h3>`;
            }
        } else {
            console.error('API error:', data.error);
            alert('Error generating result. See console for details.');
        }
    })
    .catch(error => {
        console.error('API request failed:', error);
        alert('Failed to communicate with the API.');
    });
}

// Helper function to display API status on the page
function displayApiStatus(message, isSuccess) {
    // Create or update a status element on the page
    let statusElement = document.getElementById('api-status');
    if (!statusElement) {
        statusElement = document.createElement('div');
        statusElement.id = 'api-status';
        statusElement.style.padding = '10px';
        statusElement.style.margin = '10px';
        statusElement.style.borderRadius = '5px';
        document.body.prepend(statusElement);
    }
    
    statusElement.textContent = message;
    statusElement.style.backgroundColor = isSuccess ? '#d4edda' : '#f8d7da';
    statusElement.style.color = isSuccess ? '#155724' : '#721c24';
    statusElement.style.border = isSuccess ? '1px solid #c3e6cb' : '1px solid #f5c6cb';
}