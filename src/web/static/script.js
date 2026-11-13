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
const statusNote = document.getElementById('statusNote');
const qualityPreset = document.getElementById('qualityPreset');

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
    hideNote();
});

downloadBtn.addEventListener('click', () => {
    if (processedBlob) {
        const url = URL.createObjectURL(processedBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = downloadName(processedBlob, processedIsVideo);
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
});

function fileExtension(file) {
    return file.name.split('.').pop().toLowerCase();
}

function imageFormat(file) {
    const extension = fileExtension(file);
    if (extension === 'png') return 'png';
    if (extension === 'webp') return 'webp';
    if (extension === 'gif') return 'gif';
    return 'jpeg';
}

function downloadName(blob, isVideo) {
    if (isVideo || blob.type.startsWith('video/')) return 'madurified.mp4';
    if (blob.type === 'image/gif') return 'madurified.gif';
    if (blob.type === 'image/png') return 'madurified.png';
    if (blob.type === 'image/webp') return 'madurified.webp';
    return 'madurified.jpg';
}

function handleFile(file) {
    const extension = fileExtension(file);
    const isImage = file.type.match(/^image\/(jpeg|png|gif|webp)$/) || ['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(extension);
    const isVideo = file.type.match(/^video\//) || ['mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v'].includes(extension);

    if (!isImage && !isVideo) {
        showError('Please upload a JPG, PNG, GIF or WebP image, or a supported video file.');
        return;
    }

    hideError();
    hideNote();
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
    formData.append('quality', qualityPreset.value);
    const maybeAnimated = ['gif', 'webp'].includes(fileExtension(file));
    const endpoint = isVideo
        ? '/process-video'
        : (maybeAnimated ? `/process-animation?fmt=${imageFormat(file)}` : `/process?fmt=${imageFormat(file)}`);

    try {
        let response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const contentType = response.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.detail || 'Processing failed');
            }
            if (payload.job_id) {
                response = await waitForJob(payload);
            }
        }

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
        const loadingText = loadingOverlay.querySelector('p');
        loadingText.textContent = 'Processing...';
    }
}

async function waitForJob(job) {
    const loadingText = loadingOverlay.querySelector('p');
    while (true) {
        await new Promise(resolve => setTimeout(resolve, 500));
        const statusResponse = await fetch(job.status_url);
        if (!statusResponse.ok) {
            throw new Error('Could not check processing status');
        }
        const status = await statusResponse.json();
        if (status.total > 0) {
            const percent = Math.floor(status.progress * 100 / status.total);
            loadingText.textContent = `Processing... ${percent}%`;
        }
        if (status.status === 'failed') {
            throw new Error(status.error || 'Processing failed');
        }
        if (status.status === 'completed') {
            if (status.audio_note) showNote(status.audio_note);
            return fetch(job.download_url);
        }
    }
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

function showNote(message) {
    statusNote.textContent = message;
    statusNote.classList.remove('hidden');
}

function hideNote() {
    statusNote.textContent = '';
    statusNote.classList.add('hidden');
}
