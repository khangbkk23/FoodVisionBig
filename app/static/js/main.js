document.addEventListener("DOMContentLoaded", () => {

    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("preview-image");
    const fileNameDisplay = document.getElementById("file-name");
    const dropPrompt = document.getElementById("drop-prompt");
    const clientError = document.getElementById("client-error");
    const uploadForm = document.getElementById("upload-form");
    const loadingOverlay = document.getElementById("loading-overlay");

    if (!dropZone || !fileInput || !uploadForm) return;

    const MAX_FILE_SIZE = 5 * 1024 * 1024;

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    ["dragleave", "drop"].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();

        if (e.dataTransfer.files.length > 1) {
            showError("Chỉ có thể tải lên một hình ảnh.");
            return;
        }

        const droppedFile = e.dataTransfer.files[0];

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(droppedFile);
        fileInput.files = dataTransfer.files;

        handleFile(droppedFile);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {

        clientError.style.display = "none";

        if (!["image/jpeg", "image/png"].includes(file.type)) {
            showError("Chỉ hỗ trợ ảnh JPEG hoặc PNG.");
            resetPreview();
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            showError(`Dung lượng tệp ${(file.size/1024/1024).toFixed(2)} MB vượt quá 5MB.`);
            resetPreview();
            return;
        }

        const reader = new FileReader();

        reader.onload = () => {
            previewImage.src = reader.result;
            fileNameDisplay.textContent =
                `${file.name} (${(file.size/1024).toFixed(1)} KB)`;

            dropPrompt.style.display = "none";
            previewContainer.style.display = "block";
        };

        reader.readAsDataURL(file);
    }

    function showError(message) {
        clientError.textContent = message;
        clientError.style.display = "block";
        fileInput.value = "";
    }

    function resetPreview() {
        previewContainer.style.display = "none";
        dropPrompt.style.display = "block";
        previewImage.src = "";
    }

    // Submit form
    uploadForm.addEventListener("submit", (e) => {

        if (fileInput.files.length === 0) {
            e.preventDefault();
            showError("Vui lòng tải lên một hình ảnh trước khi gửi.");
            return;
        }

        // Hiển thị spinner khi model predict
        if (loadingOverlay) {
            loadingOverlay.style.display = "flex";
        }

    });

});