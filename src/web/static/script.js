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
        a.download = processedIsVideo ? 'madurified.mp4' : 'madurified.jpg';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
});

function handleFile(file) {
    const isImage = file.type.match(/^image\/(jpeg|png)$/);
    const isVideo = file.type.match(/^video\//);

    if (!isImage && !isVideo) {
        showError('Please upload a JPG/PNG image or a video file');
        return;
    }

    hideError();
    processedIsVideo = isVideo;

    originalPreview.classList.toggle('hidden', isVideo);
    resultPreview.classList.toggle('hidden', isVideo);
    originalVideo.classList.toggle('hidden', !isVideo);
    resultVideo.classList.toggle('hidden', !isVideo);

    const localUrl = URL.createObjectURL(file);
    if (isVideo) {
        originalVideo.src = localUrl;
    } else {
        originalPreview.src = localUrl;
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
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }

        processedBlob = await response.blob();
        const url = URL.createObjectURL(processedBlob);
        if (isVideo) {
            resultVideo.src = url;
        } else {
            resultPreview.src = url;
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
