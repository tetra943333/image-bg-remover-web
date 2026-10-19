const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");
const selectBtn = document.getElementById("selectBtn");
const previewArea = document.getElementById("previewArea");

let imageFiles = [];

selectBtn.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  addFiles(e.target.files);
});

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();

  dropArea.classList.remove("dragover");

  addFiles(e.dataTransfer.files);
});

function addFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith("image/")) {
      continue;
    }

    imageFiles.push(file);

    createPreview(file);
  }
}

function createPreview(file) {
  const reader = new FileReader();

  reader.onload = function (event) {
    const card = document.createElement("div");
    card.className = "card";

    card.innerHTML = `
            <img src="${event.target.result}">
            <div class="filename">${file.name}</div>
            <button class="removeBtn">削除</button>
        `;

    card.querySelector(".removeBtn").addEventListener("click", () => {
      imageFiles = imageFiles.filter((f) => f !== file);

      card.remove();
    });

    previewArea.appendChild(card);
  };

  reader.readAsDataURL(file);
}

document.getElementById("processBtn").addEventListener("click", () => {
  alert("STEP2でAI背景削除を実装します\n現在はプレビューのみです");
});
