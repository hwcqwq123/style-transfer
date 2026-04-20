<template>
  <div class="page">
    <div class="container">
      <header class="header">
        <h1>图像风格迁移系统</h1>
        <p>上传内容图和风格图，选择算法后生成结果</p>
      </header>

      <div class="toolbar">
        <div class="left-tools">
          <div class="select-group">
            <label for="method">算法选择</label>
            <select id="method" v-model="form.method">
              <option value="adam">Adam</option>
              <option value="lbfgs">LBFGS</option>
              <option value="cyclegan">CycleGAN</option>
            </select>
          </div>
        </div>

        <div class="right-tools">
          <button class="generate-btn" :disabled="loading || !canGenerate" @click="handleGenerate">
            {{ loading ? '生成中...' : '开始生成' }}
          </button>
        </div>
      </div>

      <div class="notice-box">
        <p><strong>当前算法：</strong>{{ methodLabel }}</p>
        <p v-if="form.method === 'cyclegan'" class="tip-warning">
          提示：CycleGAN 只需要上传内容图，系统将使用已训练好的模型直接生成结果图。
        </p>
      </div>

      <div class="card-grid">
        <div
          class="card upload-card"
          :class="{ dragging: dragState.content }"
          @dragover.prevent="dragState.content = true"
          @dragleave.prevent="dragState.content = false"
          @drop.prevent="onDrop($event, 'content')"
        >
          <div class="card-title">内容图</div>
          <input
            ref="contentInputRef"
            type="file"
            class="hidden-input"
            accept="image/*"
            @change="onFileChange($event, 'content')"
          />

          <div v-if="!previews.content" class="empty-box" @click="openFileDialog('content')">
            <div class="icon">🖼️</div>
            <div>拖拽内容图到这里</div>
            <div class="sub-text">或点击上传</div>
          </div>

          <div v-else class="preview-area">
            <img :src="previews.content" alt="内容图" class="preview-image" />
            <div class="file-name">{{ files.content?.name }}</div>
            <div class="btn-row">
              <button @click="openFileDialog('content')">重新选择</button>
              <button class="danger" @click="clearFile('content')">删除</button>
            </div>
          </div>
        </div>

        <div
          v-if="form.method !== 'cyclegan'"
          class="card upload-card"
          :class="{ dragging: dragState.style }"
          @dragover.prevent="dragState.style = true"
          @dragleave.prevent="dragState.style = false"
          @drop.prevent="onDrop($event, 'style')"
        >
          <div class="card-title">风格图</div>
          <input
            ref="styleInputRef"
            type="file"
            class="hidden-input"
            accept="image/*"
            @change="onFileChange($event, 'style')"
          />

          <div v-if="!previews.style" class="empty-box" @click="openFileDialog('style')">
            <div class="icon">🎨</div>
            <div>拖拽风格图到这里</div>
            <div class="sub-text">或点击上传</div>
          </div>

          <div v-else class="preview-area">
            <img :src="previews.style" alt="风格图" class="preview-image" />
            <div class="file-name">{{ files.style?.name }}</div>
            <div class="btn-row">
              <button @click="openFileDialog('style')">重新选择</button>
              <button class="danger" @click="clearFile('style')">删除</button>
            </div>
          </div>
        </div>

        <div class="card result-card">
          <div class="card-title">生成结果</div>

          <div v-if="loading" class="loading-box">
            <div class="spinner"></div>
            <p>后端正在生成图片，请稍候...</p>
          </div>

          <div v-else-if="resultImage" class="preview-area">
            <img :src="resultImage" alt="结果图" class="preview-image" />
            <div class="btn-row">
              <a :href="resultImage" download="result.jpg">
                <button>下载结果</button>
              </a>
              <button class="danger" @click="clearResult">清空结果</button>
            </div>
          </div>

          <div v-else class="empty-box result-empty">
            <div class="icon">✨</div>
            <div>点击“开始生成”后在这里显示结果</div>
          </div>
        </div>
      </div>

      <div v-if="message.text" class="message" :class="message.type">
        {{ message.text }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue'

const API_BASE_URL = 'http://127.0.0.1:5000'

const form = reactive({
  method: 'adam',
})

const files = reactive({
  content: null,
  style: null,
})

const previews = reactive({
  content: '',
  style: '',
})

const dragState = reactive({
  content: false,
  style: false,
})

const message = reactive({
  text: '',
  type: 'info',
})

const loading = ref(false)
const resultImage = ref('')
const contentInputRef = ref(null)
const styleInputRef = ref(null)

const methodLabel = computed(() => {
  const map = {
    adam: 'Adam 风格迁移',
    lbfgs: 'LBFGS 风格迁移',
    cyclegan: 'CycleGAN 域转换',
  }
  return map[form.method]
})

const canGenerate = computed(() => {
  if (form.method === 'cyclegan') {
    return !!files.content
  }
  return !!(files.content && files.style)
})

function setMessage(text, type = 'info') {
  message.text = text
  message.type = type
}

function openFileDialog(type) {
  if (type === 'content') {
    contentInputRef.value?.click()
  } else {
    styleInputRef.value?.click()
  }
}

function onFileChange(event, type) {
  const file = event.target.files?.[0]
  if (!file) return
  updateFile(type, file)
}

function onDrop(event, type) {
  dragState[type] = false
  const file = event.dataTransfer?.files?.[0]
  if (!file) return
  updateFile(type, file)
}

function updateFile(type, file) {
  if (!file.type.startsWith('image/')) {
    setMessage('请上传图片文件', 'error')
    return
  }

  if (previews[type]) {
    URL.revokeObjectURL(previews[type])
  }

  files[type] = file
  previews[type] = URL.createObjectURL(file)
  setMessage(`${type === 'content' ? '内容图' : '风格图'}上传成功`, 'success')
}

function clearFile(type) {
  if (previews[type]) {
    URL.revokeObjectURL(previews[type])
  }

  previews[type] = ''
  files[type] = null

  if (type === 'content' && contentInputRef.value) {
    contentInputRef.value.value = ''
  }
  if (type === 'style' && styleInputRef.value) {
    styleInputRef.value.value = ''
  }
}

function clearResult() {
  resultImage.value = ''
  setMessage('结果已清空', 'info')
}

watch(
  () => form.method,
  (newMethod) => {
    if (newMethod === 'cyclegan') {
      clearFile('style')
    }
  }
)

async function handleGenerate() {
  if (!canGenerate.value) {
    if (form.method === 'cyclegan') {
      setMessage('请先上传内容图', 'error')
    } else {
      setMessage('请先上传内容图和风格图', 'error')
    }
    return
  }

  loading.value = true
  resultImage.value = ''
  setMessage('开始生成，请稍候...', 'info')

  try {
    const formData = new FormData()
    formData.append('content_image', files.content)
    formData.append('method', form.method)

    if (form.method !== 'cyclegan' && files.style) {
      formData.append('style_image', files.style)
    }

    const response = await fetch(`${API_BASE_URL}/api/style-transfer`, {
      method: 'POST',
      body: formData,
    })

    const data = await response.json()

    if (!response.ok || !data.success) {
      throw new Error(data.message || '生成失败')
    }

    resultImage.value = data.image_url.startsWith('http')
      ? data.image_url
      : `${API_BASE_URL}${data.image_url}`

    setMessage(data.message || '生成成功', 'success')
  } catch (error) {
    console.error(error)
    setMessage(error.message || '生成失败，请检查后端', 'error')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
* {
  box-sizing: border-box;
}

.page {
  min-height: 100vh;
  padding: 32px 20px;
  background: linear-gradient(135deg, #f4f7fb 0%, #edf2ff 100%);
}

.container {
  max-width: 1280px;
  margin: 0 auto;
}

.header {
  margin-bottom: 24px;
}

.header h1 {
  margin: 0 0 10px;
  color: #1f2937;
  font-size: 32px;
}

.header p {
  margin: 0;
  color: #6b7280;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  margin-bottom: 18px;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}

.select-group {
  display: flex;
  align-items: center;
  gap: 12px;
}

.select-group label {
  font-weight: bold;
  color: #374151;
}

.select-group select {
  width: 200px;
  height: 42px;
  border: 1px solid #d1d5db;
  border-radius: 10px;
  padding: 0 12px;
  background: #fff;
}

.generate-btn,
.btn-row button,
.btn-row a button {
  height: 42px;
  padding: 0 18px;
  border: none;
  border-radius: 10px;
  background: #4f46e5;
  color: white;
  font-size: 14px;
  font-weight: bold;
  cursor: pointer;
}

.generate-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.notice-box {
  margin-bottom: 18px;
  padding: 14px 18px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.9);
  color: #374151;
}

.tip-warning {
  color: #b45309;
  margin-top: 8px;
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.card {
  background: rgba(255, 255, 255, 0.96);
  border-radius: 18px;
  padding: 18px;
  min-height: 440px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
}

.card-title {
  font-size: 18px;
  font-weight: bold;
  color: #1f2937;
  margin-bottom: 14px;
}

.hidden-input {
  display: none;
}

.upload-card {
  border: 2px dashed transparent;
  transition: 0.2s;
}

.upload-card.dragging {
  border-color: #4f46e5;
  background: #eef2ff;
}

.empty-box,
.loading-box {
  min-height: 360px;
  height: calc(100% - 40px);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  border-radius: 14px;
  background: #f9fafb;
  color: #6b7280;
}

.empty-box {
  cursor: pointer;
}

.result-empty {
  cursor: default;
}

.icon {
  font-size: 48px;
  margin-bottom: 10px;
}

.sub-text {
  margin-top: 8px;
  font-size: 13px;
  color: #9ca3af;
}

.preview-area {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.preview-image {
  width: 100%;
  height: 340px;
  object-fit: contain;
  background: #f3f4f6;
  border-radius: 14px;
}

.file-name {
  font-size: 13px;
  color: #6b7280;
  word-break: break-all;
}

.btn-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.danger {
  background: #ef4444 !important;
}

.message {
  margin-top: 20px;
  padding: 14px 16px;
  border-radius: 12px;
  font-weight: 600;
}

.message.info {
  background: #e0f2fe;
  color: #075985;
}

.message.success {
  background: #dcfce7;
  color: #166534;
}

.message.error {
  background: #fee2e2;
  color: #991b1b;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #c7d2fe;
  border-top-color: #4f46e5;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 14px;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 1100px) {
  .card-grid {
    grid-template-columns: 1fr;
  }

  .toolbar {
    flex-direction: column;
    align-items: stretch;
  }

  .generate-btn {
    width: 100%;
  }
}
</style>