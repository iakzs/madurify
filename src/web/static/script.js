const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const previewSection = document.getElementById('previewSection');
const originalPreview = document.getElementById('originalPreview');
const resultPreview = document.getElementById('resultPreview');
const loadingOverlay = document.getElementById('loadingOverlay');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const errorMessage = document.getElementById('errorMessage');

let processedImageBlob = null;

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

resetBtn.addEventListener('click', () => {
    uploadSection.classList.remove('hidden');
    previewSection.classList.add('hidden');
    fileInput.value = '';
    processedImageBlob = null;
    hideError();
});

downloadBtn.addEventListener('click', () => {
    if (processedImageBlob) {
        const url = URL.createObjectURL(processedImageBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'madurified.jpg';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
});

function handleFile(file) {
    if (!file.type.match(/^image\/(jpeg|png)$/)) {
        showError('Please upload a JPG or PNG image');
        return;
    }

    hideError();
    const reader = new FileReader();
    reader.onload = (e) => {
        originalPreview.src = e.target.result;
        uploadSection.classList.add('hidden');
        previewSection.classList.remove('hidden');
        processImage(file);
    };
    reader.readAsDataURL(file);
}

async function processImage(file) {
    loadingOverlay.classList.remove('hidden');
    downloadBtn.disabled = true;
    resultPreview.src = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }

        processedImageBlob = await response.blob();
        const imageUrl = URL.createObjectURL(processedImageBlob);
        resultPreview.src = imageUrl;
        downloadBtn.disabled = false;
    } catch (error) {
        showError(error.message || 'Failed to process image');
        resultPreview.src = '';
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

