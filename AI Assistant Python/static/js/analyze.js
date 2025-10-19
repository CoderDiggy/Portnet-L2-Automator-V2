// Character Counter
const textarea = document.getElementById('incidentDescription');
const charCount = document.getElementById('charCount');

textarea.addEventListener('input', function() {
    charCount.textContent = this.value.length;
    
    // Auto-resize
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Initialize character count
if (textarea.value) {
    charCount.textContent = textarea.value.length;
}

// Load Test Case
function loadTestCase(description) {
    textarea.value = description;
    charCount.textContent = description.length;
    textarea.focus();
    
    // Scroll to form
    document.querySelector('.analysis-card').scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
    });
    
    // Trigger auto-resize
    textarea.style.height = 'auto';
    textarea.style.height = (textarea.scrollHeight) + 'px';
}

// Drag and Drop Upload
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('incidentImages');
const browseBtn = document.getElementById('browseBtn');
const imagePreview = document.getElementById('imagePreview');

browseBtn.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('click', () => fileInput.click());

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop area
['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    uploadArea.classList.add('drag-over');
}

function unhighlight() {
    uploadArea.classList.remove('drag-over');
}

// Handle dropped files
uploadArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

// Handle file input change
fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

// Handle Files Function
function handleFiles(files) {
    const validFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
    if (validFiles.length === 0) return;
    
    // Update file input
    const dt = new DataTransfer();
    validFiles.forEach(file => dt.items.add(file));
    fileInput.files = dt.files;
    
    // Display previews
    displayPreviews(validFiles);
}

// Display Image Previews
function displayPreviews(files) {
    imagePreview.innerHTML = '';
    
    files.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';
            previewItem.innerHTML = `
                <img src="${e.target.result}" alt="Preview ${index + 1}">
                <button type="button" class="remove-btn" onclick="removeImage(${index})" title="Remove image">
                    <i class="fas fa-times"></i>
                </button>
                <div class="image-info">
                    <small class="text-truncate d-block">${file.name}</small>
                    <small class="text-white-50">${(file.size / 1024).toFixed(1)} KB</small>
                </div>
            `;
            imagePreview.appendChild(previewItem);
        };
        reader.readAsDataURL(file);
    });
}

// Remove Image
function removeImage(index) {
    const dt = new DataTransfer();
    const files = Array.from(fileInput.files);
    
    files.forEach((file, i) => {
        if (i !== index) dt.items.add(file);
    });
    
    fileInput.files = dt.files;
    displayPreviews(Array.from(fileInput.files));
}

// Camera Functionality
let currentStream = null;
let capturedPhotos = [];

document.getElementById('openCameraBtn').addEventListener('click', function() {
    const modal = new bootstrap.Modal(document.getElementById('cameraModal'));
    modal.show();
    openCamera();
});

async function openCamera() {
    const video = document.getElementById('cameraVideo');
    const container = document.getElementById('cameraContainer');
    const loading = document.getElementById('cameraLoading');
    const error = document.getElementById('cameraError');
    const captureBtn = document.getElementById('capturePhotoBtn');
    
    try {
        // Reset UI
        container.style.display = 'none';
        error.classList.add('d-none');
        loading.style.display = 'block';
        captureBtn.classList.add('d-none');
        
        // Request camera access
        currentStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' }
        });
        
        video.srcObject = currentStream;
        
        video.onloadedmetadata = function() {
            loading.style.display = 'none';
            container.style.display = 'block';
            captureBtn.classList.remove('d-none');
        };
        
    } catch (err) {
        console.error('Camera access error:', err);
        loading.style.display = 'none';
        error.classList.remove('d-none');
        
        let errorMsg = 'Camera access denied or not available.';
        if (err.name === 'NotAllowedError') {
            errorMsg = 'Camera permission denied. Please allow camera access and try again.';
        } else if (err.name === 'NotFoundError') {
            errorMsg = 'No camera found on this device.';
        } else if (err.name === 'NotSupportedError') {
            errorMsg = 'Camera not supported on this browser.';
        }
        
        document.getElementById('errorMessage').textContent = errorMsg;
    }
}

document.getElementById('capturePhotoBtn').addEventListener('click', capturePhoto);
document.getElementById('retakePhotoBtn').addEventListener('click', openCamera);

function capturePhoto() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('photoCanvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(function(blob) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `camera-capture-${timestamp}.jpg`;
        const file = new File([blob], filename, { type: 'image/jpeg' });
        
        addCapturedPhotoToInput(file);
        
        const modal = bootstrap.Modal.getInstance(document.getElementById('cameraModal'));
        modal.hide();
        stopCamera();
    }, 'image/jpeg', 0.9);
}

function addCapturedPhotoToInput(file) {
    const dt = new DataTransfer();
    
    // Add existing files
    Array.from(fileInput.files).forEach(f => dt.items.add(f));
    
    // Add captured photo
    dt.items.add(file);
    
    fileInput.files = dt.files;
    displayPreviews(Array.from(fileInput.files));
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
}

document.getElementById('cameraModal').addEventListener('hidden.bs.modal', stopCamera);

// Form Submission Loading State
document.getElementById('incidentForm').addEventListener('submit', function() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const spinner = analyzeBtn.querySelector('.spinner-border');
    const btnText = analyzeBtn.querySelector('.btn-text');
    
    analyzeBtn.disabled = true;
    spinner.style.display = 'inline-block';
    btnText.textContent = 'Analyzing...';
});

// Show Error Modal from URL Parameters
document.addEventListener('DOMContentLoaded', function() {
    const urlParams = new URLSearchParams(window.location.search);
    const error = urlParams.get('error');
    
    if (error) {
        const modal = new bootstrap.Modal(document.getElementById('invalidInputDialog'));
        document.getElementById('invalidInputText').textContent = decodeURIComponent(error.replace(/\+/g, ' '));
        modal.show();
    }
});
