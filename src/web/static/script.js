const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const previewSection = document.getElementById('previewSection');
const originalPreview = document.getElementById('originalPreview');
const resultPreview = document.getElementById('resultPreview');
const originalVideo = document.getElementById('originalVideo');
const resultVideo = document.getElementById('resultVideo');
const loadingOverlay = document.getElementById('loadingOverlay');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const errorMessage = document.getElementById('errorMessage');

let processedBlob = null;
let processedIsVideo = false;
let originalUrl = null;
let processedUrl = null;

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
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    if (processedUrl) URL.revokeObjectURL(processedUrl);
    originalUrl = null;
    processedUrl = null;
    processedBlob = null;
    processedIsVideo = false;
    originalVideo.src = '';
    resultVideo.src = '';
    hideError();
});

downloadBtn.addEventListener('click', () => {
    if (processedBlob) {
        const url = URL.createObjectURL(processedBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = processedIsVideo ? 'madurified.mp4' : `madurified.${processedBlob.type === 'image/gif' ? 'gif' : processedBlob.type === 'image/png' ? 'png' : 'jpg'}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
});

function handleFile(file) {
    const extension = file.name.split('.').pop().toLowerCase();
    const isImage = file.type.match(/^image\/(jpeg|png|gif)$/) || ['jpg', 'jpeg', 'png', 'gif'].includes(extension);
    const isVideo = file.type.match(/^video\//) || ['mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v'].includes(extension);

    if (!isImage && !isVideo) {
        showError('Please upload a JPG, PNG or GIF image, or a supported video file.');
        return;
    }

    hideError();
    processedIsVideo = isVideo;

    originalPreview.classList.toggle('hidden', isVideo);
    resultPreview.classList.toggle('hidden', isVideo);
    originalVideo.classList.toggle('hidden', !isVideo);
    resultVideo.classList.toggle('hidden', !isVideo);

    if (originalUrl) URL.revokeObjectURL(originalUrl);
    if (processedUrl) URL.revokeObjectURL(processedUrl);
    originalUrl = URL.createObjectURL(file);
    if (isVideo) {
        originalVideo.src = originalUrl;
    } else {
        originalPreview.src = originalUrl;
    }

    uploadSection.classList.add('hidden');
    previewSection.classList.remove('hidden');
    processFile(file, isVideo);
}

async function processFile(file, isVideo) {
    loadingOverlay.classList.remove('hidden');
    downloadBtn.disabled = true;
    resultPreview.src = '';
    resultVideo.src = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(isVideo ? '/process-video' : '/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            let message = 'Processing failed';
            try {
                const error = await response.json();
                message = error.detail || message;
            } catch (_) {
                // The server may return a non-JSON error page.
            }
            throw new Error(message);
        }

        processedBlob = await response.blob();
        processedUrl = URL.createObjectURL(processedBlob);
        if (isVideo) {
            resultVideo.src = processedUrl;
        } else {
            resultPreview.src = processedUrl;
        }
        downloadBtn.disabled = false;
    } catch (error) {
        showError(error.message || 'Failed to process file');
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
