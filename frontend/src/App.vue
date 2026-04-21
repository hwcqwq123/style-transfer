<template>
  <div class="page">
    <div class="container">
      <header class="header">
        <h1>图像风格迁移系统</h1>
        <p>上传内容图和风格图，选择算法并设置参数后生成结果</p>
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

          <button
            v-if="form.method !== 'cyclegan'"
            class="param-btn"
            type="button"
            @click="openParamModal"
          >
            调整参数
          </button>
        </div>

        <div class="right-tools">
          <button class="secondary-btn" type="button" :disabled="loading" @click="resetFrontendState">
            清空页面
          </button>

          <button class="generate-btn" type="button" :disabled="loading || !canGenerate" @click="handleGenerate">
            {{ loading ? '生成中...' : '开始生成' }}
          </button>
        </div>
      </div>

      <div class="notice-box">
        <p><strong>当前算法：</strong>{{ methodLabel }}</p>

        <template v-if="form.method === 'adam'">
          <p><strong>当前参数：</strong>{{ adamParamSummary }}</p>
        </template>

        <template v-else-if="form.method === 'lbfgs'">
          <p><strong>当前参数：</strong>{{ lbfgsParamSummary }}</p>
        </template>

        <p v-if="form.method === 'cyclegan'" class="tip-warning">
          提示：CycleGAN 只需要上传内容图，不需要风格图，也不需要调这些参数。
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
            class="hidden-input"
            type="file"
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
              <button type="button" @click="openFileDialog('content')">重新选择</button>
              <button type="button" class="danger" @click="clearFile('content')">删除</button>
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
            class="hidden-input"
            type="file"
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
              <button type="button" @click="openFileDialog('style')">重新选择</button>
              <button type="button" class="danger" @click="clearFile('style')">删除</button>
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
                <button type="button">下载结果</button>
              </a>
              <button type="button" class="danger" @click="clearResult()">清空结果</button>
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

    <div v-if="showParamModal && form.method !== 'cyclegan'" class="modal-mask" @click.self="closeParamModal">
      <div class="modal-panel">
        <div class="modal-header">
          <h2>{{ form.method === 'adam' ? 'Adam 参数设置' : 'LBFGS 参数设置' }}</h2>
          <button class="close-btn" type="button" @click="closeParamModal">×</button>
        </div>

        <div class="modal-body">
          <template v-if="form.method === 'adam'">
            <div class="form-grid">
              <div class="form-item">
                <label>迭代次数 STEPS</label>
                <input v-model.number="paramDraft.adam.steps" type="number" min="1" />
              </div>

              <div class="form-item">
                <label>打印间隔 PRINT_EVERY</label>
                <input v-model.number="paramDraft.adam.print_every" type="number" min="1" />
              </div>

              <div class="form-item">
                <label>调试图保存间隔 SAVE_DEBUG_EVERY</label>
                <input v-model.number="paramDraft.adam.save_debug_every" type="number" min="1" />
              </div>

              <div class="form-item">
                <label>最大边长 MAX_SIZE</label>
                <input v-model.number="paramDraft.adam.max_size" type="number" min="64" />
              </div>

              <div class="form-item">
                <label>内容权重 CONTENT_WEIGHT</label>
                <input v-model.number="paramDraft.adam.content_weight" type="number" min="0" step="0.000001" />
              </div>

              <div class="form-item">
                <label>风格权重 STYLE_WEIGHT</label>
                <input v-model.number="paramDraft.adam.style_weight" type="number" min="0" step="0.000001" />
              </div>

              <div class="form-item">
                <label>TV 权重 TV_WEIGHT</label>
                <input v-model.number="paramDraft.adam.tv_weight" type="number" min="0" step="0.000001" />
              </div>

              <div class="form-item">
                <label>学习率 LR</label>
                <input v-model.number="paramDraft.adam.lr" type="number" min="0.000001" step="0.000001" />
              </div>
            </div>
          </template>

          <template v-else-if="form.method === 'lbfgs'">
            <div class="form-grid">
              <div class="form-item">
                <label>迭代次数 STEPS</label>
                <input v-model.number="paramDraft.lbfgs.steps" type="number" min="1" />
              </div>

              <div class="form-item">
                <label>打印间隔 PRINT_EVERY</label>
                <input v-model.number="paramDraft.lbfgs.print_every" type="number" min="1" />
              </div>

              <div class="form-item">
                <label>调试图保存间隔 SAVE_DEBUG_EVERY</label>
                <input v-model.number="paramDraft.lbfgs.save_debug_every" type="number" min="1" />
              </div>

              <div class="form-item">
                <label>最大边长 MAX_SIZE</label>
                <input v-model.number="paramDraft.lbfgs.max_size" type="number" min="64" />
              </div>

              <div class="form-item">
                <label>内容权重 CONTENT_WEIGHT</label>
                <input v-model.number="paramDraft.lbfgs.content_weight" type="number" min="0" step="0.000001" />
              </div>

              <div class="form-item">
                <label>风格权重 STYLE_WEIGHT</label>
                <input v-model.number="paramDraft.lbfgs.style_weight" type="number" min="0" step="0.000001" />
              </div>

              <div class="form-item">
                <label>TV 权重 TV_WEIGHT</label>
                <input v-model.number="paramDraft.lbfgs.tv_weight" type="number" min="0" step="0.000001" />
              </div>
            </div>
          </template>
        </div>

        <div class="modal-footer">
          <button type="button" class="secondary-btn" @click="resetCurrentMethodParams">恢复默认</button>
          <button type="button" class="secondary-btn" @click="closeParamModal">取消</button>
          <button type="button" class="generate-btn" @click="saveParams">保存参数</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue'

const API_BASE_URL = 'http://127.0.0.1:5000'

const defaultParams = {
  adam: {
    steps: 1000,
    print_every: 1,
    save_debug_every: 100,
    max_size: 384,
    content_weight: 0.5,
    style_weight: 3000000,
    tv_weight: 0.000005,
    lr: 0.02,
  },
  lbfgs: {
    steps: 1000,
    print_every: 1,
    save_debug_every: 100,
    max_size: 384,
    content_weight: 0.5,
    style_weight: 3000000,
    tv_weight: 0.00001,
  },
}

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj))
}

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
const showParamModal = ref(false)

const savedParams = reactive(deepClone(defaultParams))
const paramDraft = reactive(deepClone(defaultParams))

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

const adamParamSummary = computed(() => {
  const p = savedParams.adam
  return `steps=${p.steps}，print_every=${p.print_every}，save_debug_every=${p.save_debug_every}，max_size=${p.max_size}，content_weight=${p.content_weight}，style_weight=${p.style_weight}，tv_weight=${p.tv_weight}，lr=${p.lr}`
})

const lbfgsParamSummary = computed(() => {
  const p = savedParams.lbfgs
  return `steps=${p.steps}，print_every=${p.print_every}，save_debug_every=${p.save_debug_every}，max_size=${p.max_size}，content_weight=${p.content_weight}，style_weight=${p.style_weight}，tv_weight=${p.tv_weight}`
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

function clearResult(showMsg = true) {
  resultImage.value = ''
  if (showMsg) {
    setMessage('结果已清空', 'info')
  }
}

function resetFrontendState() {
  clearFile('content')
  clearFile('style')
  clearResult(false)
  message.text = ''
  message.type = 'info'
}

function openParamModal() {
  if (form.method === 'cyclegan') return
  paramDraft[form.method] = deepClone(savedParams[form.method])
  showParamModal.value = true
}

function closeParamModal() {
  showParamModal.value = false
}

function validateMethodParams(method, params) {
  const integerFields = ['steps', 'print_every', 'save_debug_every', 'max_size']
  const floatFields =
    method === 'adam'
      ? ['content_weight', 'style_weight', 'tv_weight', 'lr']
      : ['content_weight', 'style_weight', 'tv_weight']

  for (const key of integerFields) {
    if (!Number.isFinite(params[key]) || params[key] <= 0 || !Number.isInteger(params[key])) {
      return `${key} 必须是大于 0 的整数`
    }
  }

  for (const key of floatFields) {
    if (!Number.isFinite(params[key]) || params[key] < 0) {
      return `${key} 必须是大于等于 0 的数字`
    }
  }

  if (method === 'adam' && params.lr <= 0) {
    return 'lr 必须大于 0'
  }

  return ''
}

function saveParams() {
  const currentMethod = form.method
  const err = validateMethodParams(currentMethod, paramDraft[currentMethod])
  if (err) {
    setMessage(err, 'error')
    return
  }

  savedParams[currentMethod] = deepClone(paramDraft[currentMethod])
  console.log('saved params =>', currentMethod, savedParams[currentMethod])
  showParamModal.value = false
  setMessage(`${currentMethod.toUpperCase()} 参数已保存`, 'success')
}

function resetCurrentMethodParams() {
  if (form.method === 'cyclegan') return
  paramDraft[form.method] = deepClone(defaultParams[form.method])
}

watch(
  () => form.method,
  (newMethod) => {
    if (newMethod === 'cyclegan') {
      clearFile('style')
      showParamModal.value = false
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

    if (form.method === 'adam' || form.method === 'lbfgs') {
      const paramsPayload = deepClone(savedParams[form.method])
      console.log('submit params =>', form.method, paramsPayload)
      formData.append('params', JSON.stringify(paramsPayload))
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

.left-tools,
.right-tools {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
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
.param-btn,
.secondary-btn,
.btn-row button,
.btn-row a button,
.close-btn {
  height: 42px;
  padding: 0 18px;
  border: none;
  border-radius: 10px;
  color: white;
  font-size: 14px;
  font-weight: bold;
  cursor: pointer;
}

.generate-btn {
  background: #4f46e5;
}

.param-btn {
  background: #0f766e;
}

.secondary-btn {
  background: #6b7280;
}

.close-btn {
  width: 42px;
  padding: 0;
  background: #ef4444;
  font-size: 22px;
  line-height: 1;
}

.generate-btn:disabled,
.secondary-btn:disabled {
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

.modal-mask {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.45);
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
  z-index: 999;
}

.modal-panel {
  width: min(820px, 100%);
  background: white;
  border-radius: 20px;
  box-shadow: 0 18px 48px rgba(15, 23, 42, 0.25);
  overflow: hidden;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 18px 22px;
  border-bottom: 1px solid #e5e7eb;
}

.modal-header h2 {
  margin: 0;
  color: #111827;
  font-size: 22px;
}

.modal-body {
  padding: 22px;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 18px;
}

.form-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-item label {
  font-weight: 600;
  color: #374151;
}

.form-item input {
  height: 42px;
  border: 1px solid #d1d5db;
  border-radius: 10px;
  padding: 0 12px;
  font-size: 14px;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 18px 22px 22px;
  border-top: 1px solid #e5e7eb;
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

  .left-tools,
  .right-tools {
    width: 100%;
  }

  .generate-btn {
    width: 100%;
  }
}

@media (max-width: 768px) {
  .form-grid {
    grid-template-columns: 1fr;
  }

  .modal-footer {
    flex-direction: column;
  }

  .modal-footer button {
    width: 100%;
  }
}
</style>