import {
    AutoModel,
    AutoProcessor,
    RawImage,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2";

const MODEL_ID = "Xenova/modnet";

const DEFAULT_TRANSPARENT_THRESHOLD = 40;
const DEFAULT_OPAQUE_THRESHOLD = 220;

let thresholdUpdateTimer = null;
let isReapplyingThresholds = false;

// 同時処理ではなく1枚ずつ処理し、
// スマートフォンでのメモリ不足を起こしにくくします。
const SUPPORTED_MIME_PREFIX = "image/";

const elements = {
    dropArea:
        document.getElementById("dropArea"),

    fileInput:
        document.getElementById("fileInput"),

    selectBtn:
        document.getElementById("selectBtn"),

    clearBtn:
        document.getElementById("clearBtn"),

    processAllBtn:
        document.getElementById("processAllBtn"),

    previewArea:
        document.getElementById("previewArea"),

    emptyState:
        document.getElementById("emptyState"),

    fileCount:
        document.getElementById("fileCount"),

    messageArea:
        document.getElementById("messageArea"),

    modelStatus:
        document.getElementById("modelStatus"),

    modelStatusIcon:
        document.getElementById("modelStatusIcon"),

    modelProgressArea:
        document.getElementById("modelProgressArea"),

    modelProgressBar:
        document.getElementById("modelProgressBar"),

    modelProgressText:
        document.getElementById("modelProgressText"),

    transparentThreshold:
        document.getElementById(
            "transparentThreshold"
        ),

    transparentThresholdValue:
        document.getElementById(
            "transparentThresholdValue"
        ),

    opaqueThreshold:
        document.getElementById(
            "opaqueThreshold"
        ),

    opaqueThresholdValue:
        document.getElementById(
            "opaqueThresholdValue"
        ),

    resetThresholdBtn:
        document.getElementById(
            "resetThresholdBtn"
        )
};
let imageItems = [];

let model = null;
let processor = null;

let modelLoadingPromise = null;
let isProcessingAll = false;

// モデルデータのブラウザキャッシュを有効化
env.useBrowserCache = true;

// ローカルモデルは使用せず、Hugging Faceから取得
env.allowLocalModels = false;

/**
 * ページ初期化
 */
function initialize() {
    registerEvents();

    updateThresholdDisplay();

    updateInterface();
}

/**
 * イベント登録
 */
function registerEvents() {
    elements.selectBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        elements.fileInput.click();
    });

    elements.dropArea.addEventListener("click", (event) => {
        if (event.target.closest("button")) {
            return;
        }

        elements.fileInput.click();
    });

    elements.dropArea.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            elements.fileInput.click();
        }
    });

    elements.fileInput.addEventListener("change", () => {
        addFiles(elements.fileInput.files);

        // 同じファイルをもう一度選べるようにリセット
        elements.fileInput.value = "";
    });

    elements.dropArea.addEventListener("dragenter", (event) => {
        event.preventDefault();
        elements.dropArea.classList.add("dragover");
    });

    elements.dropArea.addEventListener("dragover", (event) => {
        event.preventDefault();

        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = "copy";
        }

        elements.dropArea.classList.add("dragover");
    });

    elements.dropArea.addEventListener("dragleave", (event) => {
        event.preventDefault();

        if (
            event.relatedTarget &&
            elements.dropArea.contains(event.relatedTarget)
        ) {
            return;
        }

        elements.dropArea.classList.remove("dragover");
    });

    elements.dropArea.addEventListener("drop", (event) => {
        event.preventDefault();

        elements.dropArea.classList.remove("dragover");

        if (!event.dataTransfer) {
            return;
        }

        addFiles(event.dataTransfer.files);
    });

    elements.clearBtn.addEventListener("click", clearAllImages);

    elements.processAllBtn.addEventListener(
        "click",
        processAllImages
    );

    window.addEventListener("beforeunload", cleanupObjectUrls);

    elements.transparentThreshold.addEventListener(
    "input",
    handleThresholdInput
);
    
    elements.opaqueThreshold.addEventListener(
        "input",
        handleThresholdInput
    );
    
    elements.resetThresholdBtn.addEventListener(
        "click",
        resetThresholds
);
}

/**
 * ファイル追加
 */
function addFiles(fileList) {
    const files = Array.from(fileList);

    let addedCount = 0;
    let skippedCount = 0;

    for (const file of files) {
        if (!file.type.startsWith(SUPPORTED_MIME_PREFIX)) {
            skippedCount++;
            continue;
        }

        const duplicate = imageItems.some((item) => {
            return (
                item.file.name === file.name &&
                item.file.size === file.size &&
                item.file.lastModified === file.lastModified
            );
        });

        if (duplicate) {
            skippedCount++;
            continue;
        }

        const item = {
            id: createItemId(),
        
            file,
        
            originalUrl:
                URL.createObjectURL(file),
        
            resultUrl: null,
            resultBlob: null,
        
            /*
             * AI推論後のマスクを保存します。
             * しきい値変更時は、このマスクから
             * PNGだけを再生成します。
             */
            maskData: null,
            maskWidth: 0,
            maskHeight: 0,
        
            status: "ready",
            errorMessage: ""
        };

        imageItems.push(item);
        addedCount++;
    }

    renderAllCards();
    updateInterface();

    if (addedCount > 0) {
        showMessage(
            `${addedCount}枚の画像を追加しました。`,
            "success"
        );
    }

    if (skippedCount > 0) {
        showMessage(
            `${skippedCount}件は画像以外、または重複のため追加しませんでした。`,
            "info"
        );
    }
}

/**
 * 一意ID作成
 */
function createItemId() {
    if (
        typeof crypto !== "undefined" &&
        typeof crypto.randomUUID === "function"
    ) {
        return crypto.randomUUID();
    }

    return (
        Date.now().toString(36) +
        Math.random().toString(36).slice(2)
    );
}

/**
 * 全カード再描画
 */
function renderAllCards() {
    elements.previewArea.innerHTML = "";

    for (const item of imageItems) {
        const card = createImageCard(item);
        elements.previewArea.appendChild(card);
    }
}

/**
 * 画像カード作成
 */
function createImageCard(item) {
    const card = document.createElement("article");

    card.className = "image-card";
    card.dataset.itemId = item.id;

    const stage = document.createElement("div");
    stage.className = "image-stage";

    const image = document.createElement("img");
    image.className = "preview-image";
    image.alt = item.file.name;
    image.src = item.resultUrl || item.originalUrl;

    stage.appendChild(image);

    if (item.status === "processing") {
        const overlay = document.createElement("div");
        overlay.className = "card-overlay";

        const spinner = document.createElement("div");
        spinner.className = "spinner";
        spinner.setAttribute("aria-hidden", "true");

        const processingText = document.createElement("div");
        processingText.textContent = "背景を削除しています...";

        overlay.appendChild(spinner);
        overlay.appendChild(processingText);
        stage.appendChild(overlay);
    }

    const content = document.createElement("div");
    content.className = "card-content";

    const filename = document.createElement("div");
    filename.className = "filename";
    filename.textContent = item.file.name;

    const detail = document.createElement("div");
    detail.className = "file-detail";
    detail.textContent = formatBytes(item.file.size);

    const status = document.createElement("div");
    status.className = "card-status";

    if (item.status === "completed") {
        status.classList.add("success");
        status.textContent = "背景削除完了";
    } else if (item.status === "processing") {
        status.textContent = "処理中...";
    } else if (item.status === "error") {
        status.classList.add("error");
        status.textContent =
            item.errorMessage || "処理に失敗しました";
    } else {
        status.textContent = "未処理";
    }

    const buttonArea = document.createElement("div");
    buttonArea.className = "card-buttons";

    const processButton = document.createElement("button");
    processButton.type = "button";
    processButton.className =
        "button button-success button-small";

    processButton.textContent =
        item.status === "completed"
            ? "再処理"
            : "背景削除";

    processButton.disabled =
        item.status === "processing" ||
        isProcessingAll;

    processButton.addEventListener("click", async () => {
        await processSingleImage(item.id);
    });

    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className =
        "button button-danger button-small";

    removeButton.textContent = "削除";

    removeButton.disabled =
        item.status === "processing" ||
        isProcessingAll;

    removeButton.addEventListener("click", () => {
        removeImage(item.id);
    });

    buttonArea.appendChild(processButton);
    buttonArea.appendChild(removeButton);

    if (item.status === "completed" && item.resultUrl) {
        const downloadButton = document.createElement("button");

        downloadButton.type = "button";

        downloadButton.className =
            "button button-primary button-small download-button";

        downloadButton.textContent = "透過PNGを保存";

        downloadButton.addEventListener("click", () => {
            downloadResult(item.id);
        });

        buttonArea.appendChild(downloadButton);
    }

    content.appendChild(filename);
    content.appendChild(detail);
    content.appendChild(status);
    content.appendChild(buttonArea);

    card.appendChild(stage);
    card.appendChild(content);

    return card;
}

/**
 * AIモデル読み込み
 */
async function loadModel() {
    if (model && processor) {
        return;
    }

    if (modelLoadingPromise) {
        return modelLoadingPromise;
    }

    setModelState(
        "loading",
        "AIモデルを読み込んでいます..."
    );

    elements.modelProgressArea.classList.remove("hidden");

    modelLoadingPromise = (async () => {
        try {
            processor = await AutoProcessor.from_pretrained(
                MODEL_ID,
                {
                    progress_callback: updateModelProgress
                }
            );

            model = await AutoModel.from_pretrained(
                MODEL_ID,
                {
                    dtype: "fp32",
                    progress_callback: updateModelProgress
                }
            );

            setModelState(
                "ready",
                "AIモデルの準備が完了しました"
            );

            elements.modelProgressBar.style.width = "100%";
            elements.modelProgressText.textContent =
                "読み込み完了";
        } catch (error) {
            console.error("Model loading error:", error);

            model = null;
            processor = null;
            modelLoadingPromise = null;

            setModelState(
                "error",
                "AIモデルの読み込みに失敗しました"
            );

            throw new Error(
                "AIモデルを読み込めませんでした。通信状態を確認して、もう一度実行してください。"
            );
        }
    })();

    return modelLoadingPromise;
}

/**
 * モデル読込進捗
 */
function updateModelProgress(progress) {
    if (!progress) {
        return;
    }

    let percent = null;

    if (
        typeof progress.progress === "number" &&
        Number.isFinite(progress.progress)
    ) {
        percent = Math.max(
            0,
            Math.min(100, progress.progress)
        );
    } else if (
        typeof progress.loaded === "number" &&
        typeof progress.total === "number" &&
        progress.total > 0
    ) {
        percent = Math.max(
            0,
            Math.min(
                100,
                (progress.loaded / progress.total) * 100
            )
        );
    }

    if (percent !== null) {
        elements.modelProgressBar.style.width =
            `${percent.toFixed(1)}%`;
    }

    if (progress.status === "progress") {
        const filename =
            progress.file || "モデルデータ";

        const percentText =
            percent === null
                ? ""
                : ` ${percent.toFixed(0)}%`;

        elements.modelProgressText.textContent =
            `${filename} を読み込み中${percentText}`;
    } else if (progress.status === "done") {
        elements.modelProgressText.textContent =
            "モデルデータを確認中...";
    } else {
        elements.modelProgressText.textContent =
            "AIモデルを準備中...";
    }
}

/**
 * モデル状態表示
 */
function setModelState(state, message) {
    elements.modelStatus.textContent = message;

    elements.modelStatusIcon.className = "status-icon";

    if (state === "loading") {
        elements.modelStatusIcon.classList.add(
            "status-loading"
        );
    } else if (state === "ready") {
        elements.modelStatusIcon.classList.add(
            "status-ready"
        );
    } else if (state === "error") {
        elements.modelStatusIcon.classList.add(
            "status-error"
        );
    } else {
        elements.modelStatusIcon.classList.add(
            "status-waiting"
        );
    }
}

/**
 * 1枚処理
 */
async function processSingleImage(itemId) {
    const item = imageItems.find(
        (entry) => entry.id === itemId
    );

    if (!item || item.status === "processing") {
        return;
    }

    item.status = "processing";
    item.errorMessage = "";

    renderAllCards();
    updateInterface();

    try {
        await loadModel();

        const resultBlob = await removeBackground(item);

        if (item.resultUrl) {
            URL.revokeObjectURL(item.resultUrl);
        }

        item.resultBlob = resultBlob;
        item.resultUrl = URL.createObjectURL(resultBlob);
        item.status = "completed";

        showMessage(
            `${item.file.name} の背景削除が完了しました。`,
            "success"
        );
    } catch (error) {
        console.error("Background removal error:", error);

        item.status = "error";
        item.errorMessage =
            error instanceof Error
                ? error.message
                : "背景削除に失敗しました";

        showMessage(
            `${item.file.name}: ${item.errorMessage}`,
            "error"
        );
    }

    renderAllCards();
    updateInterface();
}

/**
 * すべて処理
 */
async function processAllImages() {
    if (isProcessingAll || imageItems.length === 0) {
        return;
    }

    isProcessingAll = true;
    updateInterface();
    renderAllCards();

    let completedCount = 0;
    let failedCount = 0;

    try {
        await loadModel();

        for (const item of imageItems) {
            item.status = "processing";
            item.errorMessage = "";

            renderAllCards();

            try {
                const resultBlob =
                    await removeBackground(item);

                if (item.resultUrl) {
                    URL.revokeObjectURL(item.resultUrl);
                }

                item.resultBlob = resultBlob;
                item.resultUrl =
                    URL.createObjectURL(resultBlob);

                item.status = "completed";
                completedCount++;
            } catch (error) {
                console.error(
                    `Processing failed: ${item.file.name}`,
                    error
                );

                item.status = "error";

                item.errorMessage =
                    error instanceof Error
                        ? error.message
                        : "背景削除に失敗しました";

                failedCount++;
            }

            renderAllCards();

            // UI描画のために処理を一度ブラウザへ返す
            await waitForNextFrame();
        }

        if (failedCount === 0) {
            showMessage(
                `${completedCount}枚の背景削除が完了しました。`,
                "success"
            );
        } else {
            showMessage(
                `${completedCount}枚完了、${failedCount}枚失敗しました。`,
                "error"
            );
        }
    } catch (error) {
        console.error("Batch processing error:", error);

        showMessage(
            error instanceof Error
                ? error.message
                : "一括処理を開始できませんでした。",
            "error"
        );
    } finally {
        isProcessingAll = false;
        updateInterface();
        renderAllCards();
    }
}

/**
 * 実際の背景削除
 */
async function removeBackground(item) {
    if (!model || !processor) {
        throw new Error(
            "AIモデルが準備されていません。"
        );
    }

    const sourceUrl =
        URL.createObjectURL(item.file);

    try {
        const originalImage =
            await RawImage.fromURL(sourceUrl);

        const modelInputs =
            await processor(originalImage);

        const modelOutput = await model({
            input: modelInputs.pixel_values
        });

        const outputTensor =
            getOutputTensor(modelOutput);

        if (!outputTensor) {
            throw new Error(
                "AIモデルからマスク画像を取得できませんでした。"
            );
        }

        /*
         * MODNetの出力は既に0～1なので、
         * sigmoidは実行しません。
         */
        let maskTensor = outputTensor;

        if (
            Array.isArray(maskTensor.dims) &&
            maskTensor.dims.length === 4
        ) {
            maskTensor =
                maskTensor.squeeze(0);
        }

        if (maskTensor.dims.length !== 3) {
            throw new Error(
                `マスクの次元が不正です: ${
                    JSON.stringify(maskTensor.dims)
                }`
            );
        }

        const channels =
            maskTensor.dims[0];

        const maskHeight =
            maskTensor.dims[1];

        const maskWidth =
            maskTensor.dims[2];

        if (channels !== 1) {
            throw new Error(
                `マスクのチャンネル数が不正です: ${
                    channels
                }`
            );
        }

        const floatMaskData =
            maskTensor.data;

        const uint8MaskData =
            new Uint8ClampedArray(
                floatMaskData.length
            );

        for (
            let index = 0;
            index < floatMaskData.length;
            index++
        ) {
            const value =
                floatMaskData[index];

            uint8MaskData[index] =
                Math.round(
                    Math.max(
                        0,
                        Math.min(1, value)
                    ) * 255
                );
        }

        /*
         * AIの元マスクを保持します。
         * しきい値処理前のデータです。
         */
        item.maskData =
            new Uint8ClampedArray(
                uint8MaskData
            );

        console.log(
            "stored mask",
            Math.min(...uint8MaskData.slice(0,1000)),
            Math.max(...uint8MaskData.slice(0,1000))
        );

        item.maskWidth = maskWidth;
        item.maskHeight = maskHeight;

        const maskImage = new RawImage(
            uint8MaskData,
            maskWidth,
            maskHeight,
            1
        );

        const resizedMask =
            await maskImage.resize(
                originalImage.width,
                originalImage.height
            );

        return await composeTransparentPng(
            item.file,
            resizedMask
        );
    } finally {
        URL.revokeObjectURL(sourceUrl);
    }
    console.log(
        "maskData",
        item.maskData?.length
    );
}

/**
 * モデル出力からTensor取得
 */
function getOutputTensor(modelOutput) {
    if (!modelOutput) {
        return null;
    }

    if (modelOutput.output) {
        return modelOutput.output;
    }

    if (modelOutput.logits) {
        return modelOutput.logits;
    }

    if (modelOutput.pred) {
        return modelOutput.pred;
    }

    if (Array.isArray(modelOutput) && modelOutput.length > 0) {
        return modelOutput[0];
    }

    const values = Object.values(modelOutput);

    return values.find((value) => {
        return (
            value &&
            typeof value === "object" &&
            typeof value.squeeze === "function"
        );
    }) || null;
}



/**
 * 元画像とマスクを合成して透過PNGを作成
 */
async function composeTransparentPng(
    file,
    maskImage
) {
    const originalBitmap =
        await createImageBitmap(file);

    const width =
        originalBitmap.width;

    const height =
        originalBitmap.height;

    const outputCanvas =
        document.createElement("canvas");

    outputCanvas.width = width;
    outputCanvas.height = height;

    const outputContext =
        outputCanvas.getContext(
            "2d",
            {
                willReadFrequently: true
            }
        );

    if (!outputContext) {
        originalBitmap.close();

        throw new Error(
            "画像処理用キャンバスを作成できませんでした。"
        );
    }

    outputContext.clearRect(
        0,
        0,
        width,
        height
    );

    outputContext.drawImage(
        originalBitmap,
        0,
        0,
        width,
        height
    );

    originalBitmap.close();

    const originalPixels =
        outputContext.getImageData(
            0,
            0,
            width,
            height
        );

    let maskData =
    normalizeMaskData(
        maskImage,
        width,
        height
    );

maskData =
    keepLargestComponentOnly(
        maskData,
        width,
        height
    );

    const originalData =
        originalPixels.data;

    const transparentThreshold =
        getTransparentThreshold();

    const opaqueThreshold =
        getOpaqueThreshold();

    let minAlpha = 255;
    let maxAlpha = 0;
    
    for(let i=0;i<maskData.length;i+=4){
    
        minAlpha = Math.min(
            minAlpha,
            maskData[i]
        );
    
        maxAlpha = Math.max(
            maxAlpha,
            maskData[i]
        );
    }

console.log(
    "mask range",
    minAlpha,
    maxAlpha
);
    
    console.log(
        "transparentThreshold",
        transparentThreshold
    );

    console.log(
        "opaqueThreshold",
        opaqueThreshold
    );
    
    for (
        let index = 0;
        index < originalData.length;
        index += 4
    ) {
        let alpha = maskData[index];

        /*
         * 下限未満は完全透明
         */
        if (alpha <= transparentThreshold) {
            alpha = 0;
        }

        /*
         * 上限以上は完全不透明
         */
        else if (alpha >= opaqueThreshold) {
            alpha = 255;
        }

        /*
         * 中間領域を0～255へ再配置。
         * 輪郭の滑らかさを維持しながら
         * 背景側の薄い残りを除去します。
         */
        else {
            alpha =
                (
                    alpha -
                    transparentThreshold
                ) /
                (
                    opaqueThreshold -
                    transparentThreshold
                );

            /*
             * Smoothstepで輪郭の急激な
             * ギザギザを抑制します。
             */
            alpha =
                alpha *
                alpha *
                (3 - 2 * alpha);

            alpha =
                Math.round(alpha * 255);
        }

        /*
         * 元画像がもともと透明な場合も維持
         */
        originalData[index + 3] =
            Math.round(
                (
                    originalData[index + 3] *
                    alpha
                ) / 255
            );
    }

    outputContext.putImageData(
        originalPixels,
        0,
        0
    );

    return await canvasToBlob(
        outputCanvas
    );
}

/**
 * RawImageのマスクをRGBAへ変換
 */
function normalizeMaskData(maskImage, width, height) {
    const expectedPixels = width * height;
    const source = maskImage.data;
    const channels = maskImage.channels || 1;

    if (!source || source.length === 0) {
        throw new Error(
            "背景マスクの画像データが空です。"
        );
    }

    const rgba = new Uint8ClampedArray(
        expectedPixels * 4
    );

    for (let pixelIndex = 0;
        pixelIndex < expectedPixels;
        pixelIndex++
    ) {
        const sourceIndex = pixelIndex * channels;

        let maskValue;

        if (channels === 1) {
            maskValue = source[pixelIndex];
        } else {
            maskValue = source[sourceIndex];
        }

        if (
            typeof maskValue !== "number" ||
            Number.isNaN(maskValue)
        ) {
            maskValue = 0;
        }

        // 0～1形式なら0～255へ変換
        if (maskValue >= 0 && maskValue <= 1) {
            maskValue *= 255;
        }

        const alpha = Math.max(
            0,
            Math.min(255, Math.round(maskValue))
        );

        const rgbaIndex = pixelIndex * 4;

        rgba[rgbaIndex] = alpha;
        rgba[rgbaIndex + 1] = alpha;
        rgba[rgbaIndex + 2] = alpha;
        rgba[rgbaIndex + 3] = 255;
    }

    return rgba;
}

function keepLargestComponentOnly(
    rgbaMask,
    width,
    height
) {
    const visited =
        new Uint8Array(width * height);

    let largestPixels = [];
    let largestSize = 0;

    const directions = [
        [-1,0],
        [1,0],
        [0,-1],
        [0,1]
    ];

    for (
        let y = 0;
        y < height;
        y++
    ) {
        for (
            let x = 0;
            x < width;
            x++
        ) {
            const index =
                y * width + x;

            if (visited[index]) {
                continue;
            }

            const alpha =
                rgbaMask[index * 4];

            if (alpha < 128) {
                continue;
            }

            const queue = [index];
            const component = [];

            visited[index] = 1;

            while (
                queue.length > 0
            ) {
                const current =
                    queue.pop();

                component.push(current);

                const cy =
                    Math.floor(
                        current / width
                    );

                const cx =
                    current % width;

                for (
                    const [dx,dy]
                    of directions
                ) {
                    const nx =
                        cx + dx;

                    const ny =
                        cy + dy;

                    if (
                        nx < 0 ||
                        ny < 0 ||
                        nx >= width ||
                        ny >= height
                    ) {
                        continue;
                    }

                    const n =
                        ny * width + nx;

                    if (
                        visited[n]
                    ) {
                        continue;
                    }

                    const neighborAlpha =
                        rgbaMask[n * 4];

                    if (
                        neighborAlpha < 128
                    ) {
                        continue;
                    }

                    visited[n] = 1;

                    queue.push(n);
                }
            }

            if (
                component.length >
                largestSize
            ) {
                largestSize =
                    component.length;

                largestPixels =
                    component;
            }
        }
    }

    console.log(
        "Largest component size:",
        largestSize
    );

    const result =
        new Uint8ClampedArray(
            rgbaMask.length
        );

    for (
        const pixelIndex
        of largestPixels
    ) {
        const i =
            pixelIndex * 4;

        result[i] =
            rgbaMask[i];

        result[i+1] =
            rgbaMask[i+1];

        result[i+2] =
            rgbaMask[i+2];

        result[i+3] =
            rgbaMask[i+3];
    }

    return result;
}

/**
 * CanvasをPNG Blobへ変換
 */
function canvasToBlob(canvas) {
    return new Promise((resolve, reject) => {
        canvas.toBlob(
            (blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(
                        new Error(
                            "透過PNGの生成に失敗しました。"
                        )
                    );
                }
            },
            "image/png"
        );
    });
}

/**
 * 透過PNGダウンロード
 */
function downloadResult(itemId) {
    const item = imageItems.find(
        (entry) => entry.id === itemId
    );

    if (!item || !item.resultUrl) {
        showMessage(
            "保存する処理結果がありません。",
            "error"
        );

        return;
    }

    const link = document.createElement("a");

    link.href = item.resultUrl;
    link.download =
        `${getFilenameWithoutExtension(item.file.name)}_transparent.png`;

    document.body.appendChild(link);
    link.click();
    link.remove();
}

/**
 * 画像削除
 */
function removeImage(itemId) {
    const targetIndex = imageItems.findIndex(
        (item) => item.id === itemId
    );

    if (targetIndex === -1) {
        return;
    }

    const item = imageItems[targetIndex];

    releaseItemUrls(item);
    imageItems.splice(targetIndex, 1);

    renderAllCards();
    updateInterface();

    showMessage(
        `${item.file.name} を一覧から削除しました。`,
        "info"
    );
}

/**
 * 全画像削除
 */
function clearAllImages() {
    if (isProcessingAll) {
        return;
    }

    for (const item of imageItems) {
        releaseItemUrls(item);
    }

    imageItems = [];

    renderAllCards();
    updateInterface();

    showMessage(
        "画像をすべて削除しました。",
        "info"
    );
}

/**
 * Object URL解放
 */
function releaseItemUrls(item) {
    if (item.originalUrl) {
        URL.revokeObjectURL(item.originalUrl);
    }

    if (item.resultUrl) {
        URL.revokeObjectURL(item.resultUrl);
    }
}

/**
 * ページ終了時URL解放
 */
function cleanupObjectUrls() {
    for (const item of imageItems) {
        releaseItemUrls(item);
    }
}

/**
 * UI状態更新
 */
function updateInterface() {
    const count = imageItems.length;
    const hasImages = count > 0;

    elements.fileCount.textContent =
        `画像 ${count}枚`;

    elements.emptyState.classList.toggle(
        "hidden",
        hasImages
    );

    elements.clearBtn.disabled =
        !hasImages || isProcessingAll;

    elements.processAllBtn.disabled =
        !hasImages || isProcessingAll;

    elements.selectBtn.disabled = isProcessingAll;

    elements.processAllBtn.textContent =
        isProcessingAll
            ? "一括処理中..."
            : "すべて背景削除";
}

/**
 * メッセージ表示
 */
function showMessage(message, type = "info") {
    elements.messageArea.textContent = message;

    elements.messageArea.className =
        `message-area message-${type}`;

    elements.messageArea.classList.remove("hidden");
}

/**
 * ファイルサイズ表示
 */
function formatBytes(bytes) {
    if (bytes === 0) {
        return "0 B";
    }

    const units = [
        "B",
        "KB",
        "MB",
        "GB"
    ];

    const unitIndex = Math.min(
        Math.floor(Math.log(bytes) / Math.log(1024)),
        units.length - 1
    );

    const value =
        bytes / Math.pow(1024, unitIndex);

    return `${value.toFixed(
        unitIndex === 0 ? 0 : 1
    )} ${units[unitIndex]}`;
}

/**
 * 拡張子を除いたファイル名
 */
function getFilenameWithoutExtension(filename) {
    const lastDot = filename.lastIndexOf(".");

    if (lastDot <= 0) {
        return filename;
    }

    return filename.slice(0, lastDot);
}

/**
 * 次の描画フレームまで待機
 */
function waitForNextFrame() {
    return new Promise((resolve) => {
        requestAnimationFrame(() => {
            resolve();
        });
    });
}

/**
 * しきい値変更
 */
function handleThresholdInput() {
    normalizeThresholdValues();
    updateThresholdDisplay();

    clearTimeout(thresholdUpdateTimer);

    /*
     * スライダー操作中に毎回PNGを作ると重いため、
     * 操作停止から少し後に再生成します。
     */
    thresholdUpdateTimer = setTimeout(() => {
        reapplyThresholdsToCompletedImages();
    }, 180);
}

/**
 * 下限が上限を超えないように調整
 */
function normalizeThresholdValues() {
    let transparentValue =
        getTransparentThreshold();

    let opaqueValue =
        getOpaqueThreshold();

    if (transparentValue >= opaqueValue) {
        opaqueValue = Math.min(
            255,
            transparentValue + 1
        );

        elements.opaqueThreshold.value =
            String(opaqueValue);
    }
}

/**
 * しきい値表示更新
 */
function updateThresholdDisplay() {
    elements.transparentThresholdValue.textContent =
        String(getTransparentThreshold());

    elements.opaqueThresholdValue.textContent =
        String(getOpaqueThreshold());
}

/**
 * 初期値に戻す
 */
function resetThresholds() {
    elements.transparentThreshold.value =
        String(DEFAULT_TRANSPARENT_THRESHOLD);

    elements.opaqueThreshold.value =
        String(DEFAULT_OPAQUE_THRESHOLD);

    updateThresholdDisplay();

    reapplyThresholdsToCompletedImages();
}

/**
 * 透明化しきい値取得
 */
function getTransparentThreshold() {

    console.log(
        "slider value =",
        elements.transparentThreshold.value
    );

    return Number(
        elements.transparentThreshold.value
    );
}

/**
 * 不透明化しきい値取得
 */
function getOpaqueThreshold() {
    return Number(
        elements.opaqueThreshold.value
    );
}

/**
 * 保存済みマスクへしきい値を再適用
 */
async function reapplyThresholdsToCompletedImages() {
    
    console.log("threshold reapplied");
    if (
        isProcessingAll ||
        isReapplyingThresholds
    ) {
        return;
    }

    const targetItems = imageItems.filter((item) => {
        return (
            item.maskData &&
            item.maskWidth > 0 &&
            item.maskHeight > 0
        );
    });

    if (targetItems.length === 0) {
        return;
    }

    isReapplyingThresholds = true;

    try {
        for (const item of targetItems) {
            item.status = "processing";

            renderAllCards();

            const resultBlob =
                await createResultFromStoredMask(item);

            if (item.resultUrl) {
                URL.revokeObjectURL(item.resultUrl);
            }

            item.resultBlob = resultBlob;
            item.resultUrl =
                URL.createObjectURL(resultBlob);

            item.status = "completed";

            renderAllCards();

            await waitForNextFrame();
        }

        showMessage(
            "しきい値を処理済み画像へ再適用しました。",
            "success"
        );
    } catch (error) {
        console.error(
            "Threshold reapply error:",
            error
        );

        showMessage(
            error instanceof Error
                ? error.message
                : "しきい値の再適用に失敗しました。",
            "error"
        );
    } finally {
        isReapplyingThresholds = false;

        updateInterface();
        renderAllCards();
    }
}

/**
 * 保存済みマスクからPNGを再生成
 */
async function createResultFromStoredMask(item) {
    if (
        !item.maskData ||
        !item.maskWidth ||
        !item.maskHeight
    ) {
        throw new Error(
            "再利用できる背景マスクがありません。"
        );
    }

    const maskImage = new RawImage(
        new Uint8ClampedArray(item.maskData),
        item.maskWidth,
        item.maskHeight,
        1
    );

    const originalBitmap =
        await createImageBitmap(item.file);

    const originalWidth =
        originalBitmap.width;

    const originalHeight =
        originalBitmap.height;

    originalBitmap.close();

    const resizedMask =
        await maskImage.resize(
            originalWidth,
            originalHeight
        );

    return await composeTransparentPng(
        item.file,
        resizedMask
    );
}

initialize();
