<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import {
  AlertTriangle,
  Bot,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  Download,
  FileText,
  Filter,
  RefreshCw,
  Save,
  Search,
  Send,
  Square,
  Mic,
  MicOff,
  Trash2,
  Volume2,
  X,
} from '@lucide/vue'

const apiBase = import.meta.env.VITE_API_BASE || ''
const activeView = ref('review')
const loading = ref(false)
const saving = ref(false)
const notice = ref('')
const networkDialog = reactive({
  visible: false,
  title: '',
  message: '',
  detail: '',
  endpoint: '',
})
const items = ref([])
const selectedItemIds = ref(new Set())
const total = ref(0)
const selected = ref(null)
const stats = ref({ total: 0, by_status: {}, by_task_type: {}, by_domain_category: {} })
const options = ref({ task_types: [], domain_categories: [], statuses: [], quality_flags: [] })
const documents = ref([])
const ragLoading = ref(false)
const ragResult = ref(null)
const ragStats = ref({ documents: 0, model: '', excludes_test_split: true })
const ragSessionId = ref(Number(localStorage.getItem('railway-rag-session-id') || 0) || null)
const ragMessages = ref([])
const imageZoom = ref(100)
const contextBooks = ref([])
const activeContextBook = ref(null)
const contextMarkdown = ref('')
const contextLoading = ref(false)
const speakingField = ref('')
const recognizingField = ref('')
const currentAudio = ref(null)
const recognitionInstance = ref(null)
const recognitionChunks = ref([])
const showLoadingOverlay = computed(() => loading.value || contextLoading.value)
const language = ref(localStorage.getItem('railway-ui-language') || 'zh')
const ragForm = reactive({
  question: '',
  top_k: 5,
  generate: true,
})
const ragExamples = [
  '中心锚结绳和固定线夹有什么要求？',
  '接触网巡视检查时应注意哪些安全要求？',
  '牵引变电所交接验收需要提交哪些资料？',
]

const languageOptions = [
  { value: 'zh', label: '中文' },
  { value: 'en', label: 'English' },
  { value: 'ms', label: 'Bahasa Malaysia' },
  { value: 'th', label: 'ไทย' },
  { value: 'vi', label: 'Tiếng Việt' },
  { value: 'kk', label: 'Қазақша' },
]

const messageText = {
  zh: {
    noSpeechText: '没有可播报的文本', playbackFailed: '语音播放失败', recognitionFailed: '语音识别失败', recorderUnsupported: '当前浏览器不支持录音上传', microphoneDenied: '无法访问麦克风', recordingFailed: '录音失败', requestFailed: '请求失败', saveSuccess: '已保存修改', createSuccess: '已新增题目，请填写问题和答案', reviewerRequired: '请先填写审核人', approvedSuccess: '已审核通过', rejectedSuccess: '已驳回', revisionSuccess: '已标记为需修改', confirmDelete: '确认删除当前候选数据？该操作只更新状态，不会物理删除数据库记录。', deletedSuccess: '已逻辑删除', imageQuestion: '请描述这张图片。', emptyMarkdown: '当前记录没有 OCR Markdown 内容。', evidence: '证据', retrievalMs: '检索', generationMs: '生成',
  },
  en: {
    noSpeechText: 'No text to play', playbackFailed: 'Speech playback failed', recognitionFailed: 'Speech recognition failed', recorderUnsupported: 'This browser does not support recording upload', microphoneDenied: 'Cannot access microphone', recordingFailed: 'Recording failed', requestFailed: 'Request failed', saveSuccess: 'Changes saved', createSuccess: 'New item created. Please enter question and answer.', reviewerRequired: 'Please enter reviewer first', approvedSuccess: 'Approved', rejectedSuccess: 'Rejected', revisionSuccess: 'Marked as needs revision', confirmDelete: 'Delete the current candidate? This only updates status and will not physically delete the database record.', deletedSuccess: 'Deleted logically', imageQuestion: 'Please describe this image.', emptyMarkdown: 'No OCR Markdown content for this record.', evidence: 'Evidence', retrievalMs: 'Retrieval', generationMs: 'Generation',
  },
  ms: {
    noSpeechText: 'Tiada teks untuk dimainkan', playbackFailed: 'Main balik suara gagal', recognitionFailed: 'Pengecaman suara gagal', recorderUnsupported: 'Pelayar ini tidak menyokong muat naik rakaman', microphoneDenied: 'Tidak dapat mengakses mikrofon', recordingFailed: 'Rakaman gagal', requestFailed: 'Permintaan gagal', saveSuccess: 'Perubahan disimpan', createSuccess: 'Item baharu dicipta. Sila isi soalan dan jawapan.', reviewerRequired: 'Sila isi penyemak dahulu', approvedSuccess: 'Diluluskan', rejectedSuccess: 'Ditolak', revisionSuccess: 'Ditanda perlu semakan', confirmDelete: 'Padam calon semasa? Ini hanya mengemas kini status dan tidak memadam rekod pangkalan data secara fizikal.', deletedSuccess: 'Dipadam secara logik', imageQuestion: 'Sila terangkan imej ini.', emptyMarkdown: 'Tiada kandungan OCR Markdown untuk rekod ini.', evidence: 'Bukti', retrievalMs: 'Carian', generationMs: 'Janaan',
  },
  th: {
    noSpeechText: 'ไม่มีข้อความให้เล่น', playbackFailed: 'เล่นเสียงไม่สำเร็จ', recognitionFailed: 'รู้จำเสียงไม่สำเร็จ', recorderUnsupported: 'เบราว์เซอร์นี้ไม่รองรับการอัปโหลดเสียงบันทึก', microphoneDenied: 'ไม่สามารถเข้าถึงไมโครโฟน', recordingFailed: 'บันทึกเสียงไม่สำเร็จ', requestFailed: 'คำขอล้มเหลว', saveSuccess: 'บันทึกการแก้ไขแล้ว', createSuccess: 'สร้างรายการใหม่แล้ว กรุณากรอกคำถามและคำตอบ', reviewerRequired: 'กรุณากรอกผู้ตรวจทานก่อน', approvedSuccess: 'อนุมัติแล้ว', rejectedSuccess: 'ปฏิเสธแล้ว', revisionSuccess: 'ทำเครื่องหมายว่าต้องแก้ไขแล้ว', confirmDelete: 'ลบรายการนี้หรือไม่? การดำเนินการนี้จะอัปเดตสถานะเท่านั้น ไม่ลบข้อมูลจริงในฐานข้อมูล', deletedSuccess: 'ลบเชิงตรรกะแล้ว', imageQuestion: 'โปรดอธิบายรูปภาพนี้', emptyMarkdown: 'ไม่มีเนื้อหา OCR Markdown สำหรับรายการนี้', evidence: 'หลักฐาน', retrievalMs: 'ค้นคืน', generationMs: 'สร้าง',
  },
  vi: {
    noSpeechText: 'Không có văn bản để phát', playbackFailed: 'Phát giọng nói thất bại', recognitionFailed: 'Nhận dạng giọng nói thất bại', recorderUnsupported: 'Trình duyệt không hỗ trợ tải bản ghi âm', microphoneDenied: 'Không thể truy cập micro', recordingFailed: 'Ghi âm thất bại', requestFailed: 'Yêu cầu thất bại', saveSuccess: 'Đã lưu thay đổi', createSuccess: 'Đã tạo mục mới. Vui lòng nhập câu hỏi và câu trả lời.', reviewerRequired: 'Vui lòng nhập người rà soát trước', approvedSuccess: 'Đã duyệt', rejectedSuccess: 'Đã từ chối', revisionSuccess: 'Đã đánh dấu cần sửa', confirmDelete: 'Xóa ứng viên hiện tại? Thao tác này chỉ cập nhật trạng thái và không xóa vật lý trong cơ sở dữ liệu.', deletedSuccess: 'Đã xóa logic', imageQuestion: 'Vui lòng mô tả hình ảnh này.', emptyMarkdown: 'Bản ghi này không có nội dung OCR Markdown.', evidence: 'Bằng chứng', retrievalMs: 'Truy xuất', generationMs: 'Sinh',
  },
  kk: {
    noSpeechText: 'Ойнататын мәтін жоқ', playbackFailed: 'Дауысты ойнату сәтсіз', recognitionFailed: 'Дауысты тану сәтсіз', recorderUnsupported: 'Бұл браузер жазбаны жүктеуді қолдамайды', microphoneDenied: 'Микрофонға қол жеткізу мүмкін емес', recordingFailed: 'Жазу сәтсіз', requestFailed: 'Сұрау сәтсіз', saveSuccess: 'Өзгерістер сақталды', createSuccess: 'Жаңа жазба жасалды. Сұрақ пен жауапты енгізіңіз.', reviewerRequired: 'Алдымен тексерушіні енгізіңіз', approvedSuccess: 'Бекітілді', rejectedSuccess: 'Қабылданбады', revisionSuccess: 'Түзету керек деп белгіленді', confirmDelete: 'Ағымдағы үміткерді жою керек пе? Бұл тек күйді жаңартады, дерекқордан физикалық жоймайды.', deletedSuccess: 'Логикалық түрде жойылды', imageQuestion: 'Осы суретті сипаттаңыз.', emptyMarkdown: 'Бұл жазбада OCR Markdown мазмұны жоқ.', evidence: 'Дәлел', retrievalMs: 'Іздеу', generationMs: 'Генерация',
  },
}

function resolveApiUrl(path) {
  if (/^(https?:|data:|blob:)/i.test(path)) return path
  return `${apiBase}${path}`
}

function networkEndpoint(path = '') {
  return resolveApiUrl(path || '/api/health') || '/api/health'
}

function showNetworkDialog(error, path = '') {
  networkDialog.visible = true
  networkDialog.title = '网络连接异常'
  networkDialog.message = '前端暂时无法连接后端服务。请检查 FRP 转发、后端进程或当前网络后再重试。'
  networkDialog.detail = error?.message || String(error || '')
  networkDialog.endpoint = networkEndpoint(path)
}

function hideNetworkDialog() {
  networkDialog.visible = false
}

function isGatewayError(status) {
  return [502, 503, 504].includes(Number(status))
}

async function checkBackendConnection() {
  try {
    const response = await fetch(networkEndpoint('/api/health'), { cache: 'no-store' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    hideNetworkDialog()
    notice.value = '后端连接已恢复'
  } catch (error) {
    showNetworkDialog(error, '/api/health')
  }
}

function stopSpeech() {
  if (currentAudio.value) {
    currentAudio.value.pause()
    currentAudio.value.currentTime = 0
    currentAudio.value = null
  }
  speakingField.value = ''
}

const zhVoice = 'zh-CN-XiaoxiaoNeural'
const enVoice = 'en-US-JennyNeural'

async function playAudioUrl(field, audioUrl, options = {}) {
  if (!audioUrl) return false
  if (options.toggle && speakingField.value === field) {
    stopSpeech()
    return true
  }
  stopSpeech()
  speakingField.value = field
  try {
    const audio = new Audio(resolveApiUrl(audioUrl))
    currentAudio.value = audio
    audio.onended = () => {
      if (speakingField.value === field) speakingField.value = ''
      if (currentAudio.value === audio) currentAudio.value = null
    }
    audio.onerror = () => {
      if (speakingField.value === field) speakingField.value = ''
      if (currentAudio.value === audio) currentAudio.value = null
      notice.value = t('playbackFailed')
    }
    await audio.play()
    return true
  } catch (error) {
    speakingField.value = ''
    currentAudio.value = null
    notice.value = error.message
    return false
  }
}

async function speakText(field, text, voice = zhVoice) {
  const content = String(text || '').trim()
  if (!content) {
    notice.value = t('noSpeechText')
    return
  }
  if (speakingField.value === field) {
    stopSpeech()
    return
  }

  stopSpeech()
  speakingField.value = field
  try {
    const response = await api('/api/tts', {
      method: 'POST',
      body: JSON.stringify({ text: content, voice, rate: 1.0 }),
    })
    const data = await response.json()
    await playAudioUrl(field, data.audio_url)
  } catch (error) {
    speakingField.value = ''
    currentAudio.value = null
    notice.value = error.message
  }
}

function stopRecognition() {
  if (recognitionInstance.value) {
    recognitionInstance.value.stop()
  }
}

function assignRecognizedText(field, text) {
  if (field === 'rag_question') ragForm.question = text
  if (field === 'question') editor.question = text
  if (field === 'answer') editor.answer = text
  if (field === 'question_en') editor.question_en = text
  if (field === 'answer_en') editor.answer_en = text
}

async function uploadSpeechForRecognition(field, blob) {
  const formData = new FormData()
  formData.append('audio', blob, 'speech.webm')
  const endpoint = '/api/asr?language=zh'
  let response
  try {
    response = await fetch(`${apiBase}${endpoint}`, {
      method: 'POST',
      body: formData,
    })
  } catch (error) {
    showNetworkDialog(error, endpoint)
    throw new Error('语音识别服务连接失败')
  }
  if (isGatewayError(response.status)) {
    showNetworkDialog(new Error(`HTTP ${response.status}`), endpoint)
    throw new Error('语音识别服务暂时不可用')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `${t('recognitionFailed')}：${response.status}`)
  }
  const data = await response.json()
  if (data.text) {
    assignRecognizedText(field, data.text)
    if (field === 'rag_question') await askRag()
  }
}

async function startSpeechRecognition(field) {
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
    notice.value = t('recorderUnsupported')
    return
  }
  if (recognizingField.value === field) {
    stopRecognition()
    return
  }

  stopRecognition()
  let stream
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  } catch (_error) {
    notice.value = t('microphoneDenied')
    return
  }
  const recorderOptions = MediaRecorder.isTypeSupported('audio/webm') ? { mimeType: 'audio/webm' } : {}
  const recorder = new MediaRecorder(stream, recorderOptions)
  recognitionChunks.value = []
  recorder.ondataavailable = (event) => {
    if (event.data?.size) recognitionChunks.value.push(event.data)
  }
  recorder.onerror = () => {
    stream.getTracks().forEach((track) => track.stop())
    recognitionInstance.value = null
    recognizingField.value = ''
    notice.value = t('recordingFailed')
  }
  recorder.onstop = async () => {
    stream.getTracks().forEach((track) => track.stop())
    const blob = new Blob(recognitionChunks.value, { type: 'audio/webm' })
    recognitionInstance.value = null
    recognizingField.value = ''
    recognitionChunks.value = []
    if (!blob.size) return
    try {
      await uploadSpeechForRecognition(field, blob)
    } catch (error) {
      notice.value = error.message
    }
  }
  recognitionInstance.value = recorder
  recognizingField.value = field
  recorder.start()
}

const filters = reactive({
  page: 1,
  page_size: 30,
  status: 'pending',
  task_type: '',
  domain_category: '',
  document_id: '',
  search: '',
})
const pageSizeOptions = [10, 20, 30, 50, 100, 200]

const editor = reactive({
  task_type: '',
  domain_category: '',
  knowledge_category: '',
  question: '',
  answer: '',
  question_en: '',
  answer_en: '',
  evidence: '',
  source_text: '',
  chapter: '',
  page_number: null,
  quality_flags: [],
  reviewer: localStorage.getItem('railway-reviewer') || '',
  review_comment: '',
})

const uiText = {
  zh: {
    loading: '加载中', appTitle: '铁道教育语料审核台', appSubtitle: '术语、规章、教材原文与问答候选统一校正', reviewer: '审核人', reviewerPlaceholder: '姓名或账号', language: '语言',
    review: '语料审核', rag: 'RAG 问答', context: '教材上下文', refresh: '刷新', create: '新增题目', exportApproved: '导出已通过',
    total: '全部', pending: '待审核', needs_revision: '需修改', approved: '已通过', rejected: '已驳回', candidateQueue: '候选队列', reset: '重置',
    searchPlaceholder: '问题、答案、证据或文档', allStatus: '全部状态', allTasks: '全部任务', allDomains: '全部专业', allDocuments: '全部文档', records: '条', pageSize: '每页', page: '第', noData: '当前筛选条件下没有数据', previousPage: '上一页', nextPage: '下一页',
    editorTitle: '校正与审核', previousRecord: '上一条', nextRecord: '下一条', taskType: '任务类型', domainCategory: '专业类别', knowledgeCategory: '知识类别', knowledgePlaceholder: '例如：接触网检修、牵引变电、供电原理', chapter: '章节', chapterPlaceholder: '教材章节或规章条目', question: '问题', questionZh: '问题中文', questionEn: '问题英文', questionPlaceholder: '问题必须明确包含对象和条件', questionEnPlaceholder: 'English question translated with railway terminology alignment', answer: '答案', answerZh: '答案中文', answerEn: '答案英文', answerPlaceholder: '答案必须由证据支持', answerEnPlaceholder: 'English answer translated with railway terminology alignment', playQuestion: '播放问题', playQuestionZh: '播放问题中文', playQuestionEn: '播放问题英文', playAnswer: '播放答案', playAnswerZh: '播放答案中文', playAnswerEn: '播放答案英文', stopSpeech: '停止播报', qualityFlags: '质量标记', pageNumber: '页码', reviewComment: '审核备注', save: '保存', delete: '删除', pass: '通过', selectItem: '请选择一条语料', unrecorded: '未记录', unclassified: '未分类', pendingLabel: '待标注',
    ocrPreview: 'OCR 来源预览', document: '文档', sourceType: '来源类型', ocrImage: 'OCR 原图', zoom: '缩放', ocrTextPreview: 'OCR 图文预览', rawOcrMarkdown: '查看原始 OCR Markdown', sourceContext: '原文上下文', sourceContextHint: '命中行前 20 行与后 20 行', fullSource: '完整原文', matchedLine: '命中行', matchedByMetadata: '按 OCR 行号定位',
    ragTitle: '铁路知识问答', index: '索引', evidenceCount: '证据数量', answerMode: '回答模式', generatingMode: 'Qwen3 生成', retrievalOnly: '仅检索原文', submitQuestion: '提交问题', generating: '正在检索并生成', sampleQuestions: '示例问题', model: '生成模型', testIsolation: '测试集隔离', enabled: '已启用', disabled: '未启用', answerAndEvidence: '回答与证据', retrievalEvidence: '检索证据', relevance: '相关度', askHint: '输入问题后，系统会检索本地铁路语料并给出带来源编号的回答。', voiceAsk: '后台识别提问', stopRecording: '停止录音', playAnswerResult: '播放回答',
    loadingRag: '正在检索语料并生成可追溯回答', modelWarmup: '本地模型首次加载可能需要十余秒', contextBooks: '本', chooseBook: '请选择教材', loadingContext: '正在加载教材上下文', pages: '页',
  },
  en: {
    loading: 'Loading', appTitle: 'Railway Education Corpus Review', appSubtitle: 'Unified review for terminology, regulations, textbook source text, and QA candidates', reviewer: 'Reviewer', reviewerPlaceholder: 'Name or account', language: 'Language',
    review: 'Corpus Review', rag: 'RAG Q&A', context: 'Textbook Context', refresh: 'Refresh', create: 'New Item', exportApproved: 'Export Approved',
    total: 'Total', pending: 'Pending', needs_revision: 'Needs revision', approved: 'Approved', rejected: 'Rejected', candidateQueue: 'Candidate Queue', reset: 'Reset',
    searchPlaceholder: 'Question, answer, evidence, or document', allStatus: 'All statuses', allTasks: 'All tasks', allDomains: 'All domains', allDocuments: 'All documents', records: 'items', pageSize: 'Per page', page: 'Page', noData: 'No data under current filters', previousPage: 'Previous page', nextPage: 'Next page',
    editorTitle: 'Correction & Review', previousRecord: 'Previous', nextRecord: 'Next', taskType: 'Task type', domainCategory: 'Domain', knowledgeCategory: 'Knowledge category', knowledgePlaceholder: 'e.g. catenary maintenance, traction substation, power supply principles', chapter: 'Chapter', chapterPlaceholder: 'Textbook chapter or regulation clause', question: 'Question', questionZh: 'Question Chinese', questionEn: 'Question English', questionPlaceholder: 'Question must clearly include object and condition', questionEnPlaceholder: 'English question translated with railway terminology alignment', answer: 'Answer', answerZh: 'Answer Chinese', answerEn: 'Answer English', answerPlaceholder: 'Answer must be supported by evidence', answerEnPlaceholder: 'English answer translated with railway terminology alignment', playQuestion: 'Play question', playQuestionZh: 'Play Chinese question', playQuestionEn: 'Play English question', playAnswer: 'Play answer', playAnswerZh: 'Play Chinese answer', playAnswerEn: 'Play English answer', stopSpeech: 'Stop', qualityFlags: 'Quality flags', pageNumber: 'Page', reviewComment: 'Review note', save: 'Save', delete: 'Delete', pass: 'Approve', selectItem: 'Select a corpus item', unrecorded: 'Not recorded', unclassified: 'Unclassified', pendingLabel: 'To label',
    ocrPreview: 'OCR Source Preview', document: 'Document', sourceType: 'Source type', ocrImage: 'OCR image', zoom: 'Zoom', ocrTextPreview: 'OCR Text Preview', rawOcrMarkdown: 'View raw OCR Markdown', sourceContext: 'Source Context', sourceContextHint: '20 lines before and after the matched line', fullSource: 'Full Source', matchedLine: 'Matched line', matchedByMetadata: 'located by OCR line number',
    ragTitle: 'Railway Knowledge Q&A', index: 'Index', evidenceCount: 'Evidence count', answerMode: 'Answer mode', generatingMode: 'Qwen3 generation', retrievalOnly: 'Source retrieval only', submitQuestion: 'Submit question', generating: 'Retrieving and generating', sampleQuestions: 'Sample questions', model: 'Generation model', testIsolation: 'Test isolation', enabled: 'Enabled', disabled: 'Disabled', answerAndEvidence: 'Answer & Evidence', retrievalEvidence: 'Retrieved Evidence', relevance: 'Relevance', askHint: 'Enter a question to retrieve local railway corpus and generate an answer with source references.', voiceAsk: 'Backend speech input', stopRecording: 'Stop recording', playAnswerResult: 'Play answer',
    loadingRag: 'Retrieving corpus and generating traceable answer', modelWarmup: 'The local model may take several seconds on first load', contextBooks: 'books', chooseBook: 'Select a textbook', loadingContext: 'Loading textbook context', pages: 'pages',
  },
  ms: {
    loading: 'Memuat', appTitle: 'Semakan Korpus Pendidikan Kereta Api', appSubtitle: 'Semakan bersatu untuk istilah, peraturan, teks buku dan calon soal jawab', reviewer: 'Penyemak', reviewerPlaceholder: 'Nama atau akaun', language: 'Bahasa',
    review: 'Semakan Korpus', rag: 'Soal Jawab RAG', context: 'Konteks Buku Teks', refresh: 'Muat semula', create: 'Item baharu', exportApproved: 'Eksport diluluskan',
    total: 'Jumlah', pending: 'Menunggu', needs_revision: 'Perlu semakan', approved: 'Diluluskan', rejected: 'Ditolak', candidateQueue: 'Barisan Calon', reset: 'Tetap semula',
    searchPlaceholder: 'Soalan, jawapan, bukti atau dokumen', allStatus: 'Semua status', allTasks: 'Semua tugasan', allDomains: 'Semua bidang', allDocuments: 'Semua dokumen', records: 'rekod', pageSize: 'Setiap halaman', page: 'Halaman', noData: 'Tiada data untuk penapis semasa', previousPage: 'Halaman sebelumnya', nextPage: 'Halaman seterusnya',
    editorTitle: 'Pembetulan & Semakan', previousRecord: 'Sebelumnya', nextRecord: 'Seterusnya', taskType: 'Jenis tugasan', domainCategory: 'Bidang', knowledgeCategory: 'Kategori pengetahuan', knowledgePlaceholder: 'cth. penyelenggaraan katenari, substesen traksi, prinsip bekalan kuasa', chapter: 'Bab', chapterPlaceholder: 'Bab buku teks atau klausa peraturan', question: 'Soalan', questionPlaceholder: 'Soalan mesti menyatakan objek dan syarat dengan jelas', answer: 'Jawapan', answerPlaceholder: 'Jawapan mesti disokong bukti', playQuestion: 'Main soalan', playAnswer: 'Main jawapan', stopSpeech: 'Henti', qualityFlags: 'Tanda kualiti', pageNumber: 'Halaman', reviewComment: 'Nota semakan', save: 'Simpan', delete: 'Padam', pass: 'Lulus', selectItem: 'Pilih satu item korpus', unrecorded: 'Tidak direkod', unclassified: 'Tidak dikelaskan', pendingLabel: 'Belum dilabel',
    ocrPreview: 'Pratonton Sumber OCR', document: 'Dokumen', sourceType: 'Jenis sumber', ocrImage: 'Imej OCR', zoom: 'Zum', ocrTextPreview: 'Pratonton Teks OCR', rawOcrMarkdown: 'Lihat Markdown OCR mentah', sourceContext: 'Konteks Sumber', sourceContextHint: '20 baris sebelum dan selepas baris padanan', fullSource: 'Sumber Penuh',
    ragTitle: 'Soal Jawab Pengetahuan Kereta Api', index: 'Indeks', evidenceCount: 'Bilangan bukti', answerMode: 'Mod jawapan', generatingMode: 'Jana Qwen3', retrievalOnly: 'Carian sumber sahaja', submitQuestion: 'Hantar soalan', generating: 'Mencari dan menjana', sampleQuestions: 'Contoh soalan', model: 'Model jana', testIsolation: 'Pengasingan ujian', enabled: 'Diaktifkan', disabled: 'Dilumpuhkan', answerAndEvidence: 'Jawapan & Bukti', retrievalEvidence: 'Bukti carian', relevance: 'Kerelevanan', askHint: 'Masukkan soalan untuk mencari korpus kereta api tempatan dan menjana jawapan bersumber.', voiceAsk: 'Input suara backend', stopRecording: 'Henti rakaman', playAnswerResult: 'Main jawapan',
    loadingRag: 'Mencari korpus dan menjana jawapan boleh jejak', modelWarmup: 'Model tempatan mungkin mengambil beberapa saat pada muatan pertama', contextBooks: 'buku', chooseBook: 'Pilih buku teks', loadingContext: 'Memuat konteks buku teks', pages: 'halaman',
  },
  th: {
    loading: 'กำลังโหลด', appTitle: 'ระบบตรวจทานคลังข้อมูลการศึกษารถไฟ', appSubtitle: 'ตรวจทานคำศัพท์ กฎระเบียบ ข้อความจากตำรา และชุดถามตอบในที่เดียว', reviewer: 'ผู้ตรวจทาน', reviewerPlaceholder: 'ชื่อหรือบัญชี', language: 'ภาษา',
    review: 'ตรวจทานคลังข้อมูล', rag: 'ถามตอบ RAG', context: 'บริบทตำรา', refresh: 'รีเฟรช', create: 'เพิ่มรายการ', exportApproved: 'ส่งออกที่อนุมัติ',
    total: 'ทั้งหมด', pending: 'รอตรวจ', needs_revision: 'ต้องแก้ไข', approved: 'อนุมัติแล้ว', rejected: 'ปฏิเสธแล้ว', candidateQueue: 'คิวรายการ', reset: 'รีเซ็ต',
    searchPlaceholder: 'คำถาม คำตอบ หลักฐาน หรือเอกสาร', allStatus: 'ทุกสถานะ', allTasks: 'ทุกงาน', allDomains: 'ทุกสาขา', allDocuments: 'ทุกเอกสาร', records: 'รายการ', pageSize: 'ต่อหน้า', page: 'หน้า', noData: 'ไม่มีข้อมูลตามตัวกรองปัจจุบัน', previousPage: 'หน้าก่อน', nextPage: 'หน้าถัดไป',
    editorTitle: 'แก้ไขและตรวจทาน', previousRecord: 'ก่อนหน้า', nextRecord: 'ถัดไป', taskType: 'ประเภทงาน', domainCategory: 'สาขา', knowledgeCategory: 'หมวดความรู้', knowledgePlaceholder: 'เช่น ซ่อมบำรุงสายสัมผัส สถานีไฟฟ้าฉุดลาก หลักการจ่ายไฟ', chapter: 'บท', chapterPlaceholder: 'บทตำราหรือข้อกำหนด', question: 'คำถาม', questionPlaceholder: 'คำถามต้องระบุวัตถุและเงื่อนไขให้ชัดเจน', answer: 'คำตอบ', answerPlaceholder: 'คำตอบต้องมีหลักฐานรองรับ', playQuestion: 'เล่นคำถาม', playAnswer: 'เล่นคำตอบ', stopSpeech: 'หยุด', qualityFlags: 'ป้ายคุณภาพ', pageNumber: 'หน้า', reviewComment: 'หมายเหตุ', save: 'บันทึก', delete: 'ลบ', pass: 'อนุมัติ', selectItem: 'เลือกรายการหนึ่ง', unrecorded: 'ไม่ได้บันทึก', unclassified: 'ไม่จัดประเภท', pendingLabel: 'รอติดป้าย',
    ocrPreview: 'ตัวอย่างแหล่ง OCR', document: 'เอกสาร', sourceType: 'ประเภทแหล่ง', ocrImage: 'ภาพ OCR', zoom: 'ซูม', ocrTextPreview: 'ตัวอย่างข้อความ OCR', rawOcrMarkdown: 'ดู Markdown OCR ดิบ', sourceContext: 'บริบทต้นฉบับ', sourceContextHint: '20 บรรทัดก่อนและหลังบรรทัดที่ตรงกัน', fullSource: 'ต้นฉบับเต็ม',
    ragTitle: 'ถามตอบความรู้รถไฟ', index: 'ดัชนี', evidenceCount: 'จำนวนหลักฐาน', answerMode: 'โหมดคำตอบ', generatingMode: 'สร้างด้วย Qwen3', retrievalOnly: 'ค้นต้นฉบับเท่านั้น', submitQuestion: 'ส่งคำถาม', generating: 'กำลังค้นและสร้าง', sampleQuestions: 'คำถามตัวอย่าง', model: 'โมเดลสร้าง', testIsolation: 'แยกชุดทดสอบ', enabled: 'เปิดใช้งาน', disabled: 'ปิดใช้งาน', answerAndEvidence: 'คำตอบและหลักฐาน', retrievalEvidence: 'หลักฐานที่ค้นได้', relevance: 'ความเกี่ยวข้อง', askHint: 'ป้อนคำถามเพื่อค้นคลังข้อมูลรถไฟในเครื่องและสร้างคำตอบพร้อมแหล่งอ้างอิง', voiceAsk: 'ถามด้วยเสียงผ่าน backend', stopRecording: 'หยุดบันทึก', playAnswerResult: 'เล่นคำตอบ',
    loadingRag: 'กำลังค้นคลังข้อมูลและสร้างคำตอบที่ตรวจสอบย้อนกลับได้', modelWarmup: 'โมเดลในเครื่องอาจใช้เวลาหลายวินาทีในการโหลดครั้งแรก', contextBooks: 'เล่ม', chooseBook: 'เลือกตำรา', loadingContext: 'กำลังโหลดบริบทตำรา', pages: 'หน้า',
  },
  vi: {
    loading: 'Đang tải', appTitle: 'Bàn rà soát ngữ liệu giáo dục đường sắt', appSubtitle: 'Rà soát thuật ngữ, quy định, văn bản giáo trình và câu hỏi ứng viên', reviewer: 'Người rà soát', reviewerPlaceholder: 'Tên hoặc tài khoản', language: 'Ngôn ngữ',
    review: 'Rà soát ngữ liệu', rag: 'Hỏi đáp RAG', context: 'Ngữ cảnh giáo trình', refresh: 'Làm mới', create: 'Thêm mục', exportApproved: 'Xuất đã duyệt',
    total: 'Tổng', pending: 'Chờ duyệt', needs_revision: 'Cần sửa', approved: 'Đã duyệt', rejected: 'Đã từ chối', candidateQueue: 'Hàng đợi ứng viên', reset: 'Đặt lại',
    searchPlaceholder: 'Câu hỏi, câu trả lời, bằng chứng hoặc tài liệu', allStatus: 'Tất cả trạng thái', allTasks: 'Tất cả nhiệm vụ', allDomains: 'Tất cả lĩnh vực', allDocuments: 'Tất cả tài liệu', records: 'mục', pageSize: 'Mỗi trang', page: 'Trang', noData: 'Không có dữ liệu theo bộ lọc hiện tại', previousPage: 'Trang trước', nextPage: 'Trang sau',
    editorTitle: 'Chỉnh sửa & rà soát', previousRecord: 'Trước', nextRecord: 'Sau', taskType: 'Loại nhiệm vụ', domainCategory: 'Lĩnh vực', knowledgeCategory: 'Loại kiến thức', knowledgePlaceholder: 'VD: bảo trì dây tiếp xúc, trạm điện kéo, nguyên lý cấp điện', chapter: 'Chương', chapterPlaceholder: 'Chương giáo trình hoặc điều khoản quy định', question: 'Câu hỏi', questionPlaceholder: 'Câu hỏi phải nêu rõ đối tượng và điều kiện', answer: 'Câu trả lời', answerPlaceholder: 'Câu trả lời phải có bằng chứng hỗ trợ', playQuestion: 'Phát câu hỏi', playAnswer: 'Phát câu trả lời', stopSpeech: 'Dừng', qualityFlags: 'Cờ chất lượng', pageNumber: 'Trang', reviewComment: 'Ghi chú rà soát', save: 'Lưu', delete: 'Xóa', pass: 'Duyệt', selectItem: 'Chọn một mục ngữ liệu', unrecorded: 'Chưa ghi', unclassified: 'Chưa phân loại', pendingLabel: 'Chờ gắn nhãn',
    ocrPreview: 'Xem trước nguồn OCR', document: 'Tài liệu', sourceType: 'Loại nguồn', ocrImage: 'Ảnh OCR', zoom: 'Thu phóng', ocrTextPreview: 'Xem trước văn bản OCR', rawOcrMarkdown: 'Xem Markdown OCR gốc', sourceContext: 'Ngữ cảnh nguồn', sourceContextHint: '20 dòng trước và sau dòng khớp', fullSource: 'Nguồn đầy đủ',
    ragTitle: 'Hỏi đáp kiến thức đường sắt', index: 'Chỉ mục', evidenceCount: 'Số bằng chứng', answerMode: 'Chế độ trả lời', generatingMode: 'Sinh bằng Qwen3', retrievalOnly: 'Chỉ truy xuất nguồn', submitQuestion: 'Gửi câu hỏi', generating: 'Đang truy xuất và sinh', sampleQuestions: 'Câu hỏi mẫu', model: 'Mô hình sinh', testIsolation: 'Tách tập kiểm thử', enabled: 'Bật', disabled: 'Tắt', answerAndEvidence: 'Câu trả lời & bằng chứng', retrievalEvidence: 'Bằng chứng truy xuất', relevance: 'Độ liên quan', askHint: 'Nhập câu hỏi để truy xuất ngữ liệu đường sắt cục bộ và tạo câu trả lời có nguồn.', voiceAsk: 'Nhập giọng nói backend', stopRecording: 'Dừng ghi âm', playAnswerResult: 'Phát câu trả lời',
    loadingRag: 'Đang truy xuất ngữ liệu và tạo câu trả lời có thể truy vết', modelWarmup: 'Mô hình cục bộ có thể mất vài giây ở lần tải đầu', contextBooks: 'sách', chooseBook: 'Chọn giáo trình', loadingContext: 'Đang tải ngữ cảnh giáo trình', pages: 'trang',
  },
  kk: {
    loading: 'Жүктелуде', appTitle: 'Теміржол білім корпусының тексеру тақтасы', appSubtitle: 'Терминдер, ережелер, оқулық мәтіні және сұрақ-жауап үміткерлерін бір жерде тексеру', reviewer: 'Тексеруші', reviewerPlaceholder: 'Аты немесе аккаунты', language: 'Тіл',
    review: 'Корпусты тексеру', rag: 'RAG сұрақ-жауап', context: 'Оқулық контексті', refresh: 'Жаңарту', create: 'Жаңа жазба', exportApproved: 'Бекітілгенді экспорттау',
    total: 'Барлығы', pending: 'Күтуде', needs_revision: 'Түзету керек', approved: 'Бекітілді', rejected: 'Қабылданбады', candidateQueue: 'Үміткерлер кезегі', reset: 'Қалпына келтіру',
    searchPlaceholder: 'Сұрақ, жауап, дәлел немесе құжат', allStatus: 'Барлық күйлер', allTasks: 'Барлық тапсырмалар', allDomains: 'Барлық салалар', allDocuments: 'Барлық құжаттар', records: 'жазба', pageSize: 'Әр бетте', page: 'Бет', noData: 'Ағымдағы сүзгі бойынша дерек жоқ', previousPage: 'Алдыңғы бет', nextPage: 'Келесі бет',
    editorTitle: 'Түзету және тексеру', previousRecord: 'Алдыңғы', nextRecord: 'Келесі', taskType: 'Тапсырма түрі', domainCategory: 'Сала', knowledgeCategory: 'Білім санаты', knowledgePlaceholder: 'мысалы: контакт желісін жөндеу, тарту қосалқы станциясы, электрмен жабдықтау қағидалары', chapter: 'Тарау', chapterPlaceholder: 'Оқулық тарауы немесе ереже тармағы', question: 'Сұрақ', questionPlaceholder: 'Сұрақ нысан мен шартты анық көрсетуі керек', answer: 'Жауап', answerPlaceholder: 'Жауап дәлелмен расталуы керек', playQuestion: 'Сұрақты ойнату', playAnswer: 'Жауапты ойнату', stopSpeech: 'Тоқтату', qualityFlags: 'Сапа белгілері', pageNumber: 'Бет', reviewComment: 'Тексеру ескертпесі', save: 'Сақтау', delete: 'Жою', pass: 'Бекіту', selectItem: 'Бір корпус жазбасын таңдаңыз', unrecorded: 'Жазылмаған', unclassified: 'Жіктелмеген', pendingLabel: 'Белгілеу керек',
    ocrPreview: 'OCR дереккөзі алдын ала қарау', document: 'Құжат', sourceType: 'Дереккөз түрі', ocrImage: 'OCR суреті', zoom: 'Масштаб', ocrTextPreview: 'OCR мәтінін қарау', rawOcrMarkdown: 'Бастапқы OCR Markdown қарау', sourceContext: 'Дереккөз контексті', sourceContextHint: 'Сәйкес жолдан бұрынғы және кейінгі 20 жол', fullSource: 'Толық дереккөз',
    ragTitle: 'Теміржол білімі сұрақ-жауап', index: 'Индекс', evidenceCount: 'Дәлел саны', answerMode: 'Жауап режимі', generatingMode: 'Qwen3 генерациясы', retrievalOnly: 'Тек дереккөз іздеу', submitQuestion: 'Сұрақ жіберу', generating: 'Іздеу және генерациялау', sampleQuestions: 'Үлгі сұрақтар', model: 'Генерация моделі', testIsolation: 'Тестті оқшаулау', enabled: 'Қосулы', disabled: 'Өшірулі', answerAndEvidence: 'Жауап және дәлел', retrievalEvidence: 'Табылған дәлел', relevance: 'Сәйкестік', askHint: 'Сұрақ енгізіңіз, жүйе жергілікті теміржол корпусынан іздеп, дереккөздері бар жауап береді.', voiceAsk: 'Backend дауыстық енгізу', stopRecording: 'Жазуды тоқтату', playAnswerResult: 'Жауапты ойнату',
    loadingRag: 'Корпус ізделіп, қадағаланатын жауап жасалуда', modelWarmup: 'Жергілікті модель алғаш жүктелгенде бірнеше секунд алуы мүмкін', contextBooks: 'кітап', chooseBook: 'Оқулық таңдаңыз', loadingContext: 'Оқулық контексті жүктелуде', pages: 'бет',
  },
}

const taskTypeLabelMap = {
  zh: {
    concept_explanation_qa: '概念说明问答', regulation_clause_qa: '规章条款问答', regulation_definition_qa: '规章定义问答', regulation_extractive_qa: '规章抽取式问答', regulation_inspection_qa: '规章检查检修问答', regulation_judgment: '规章判断题', regulation_principle_qa: '规章原则问答', regulation_prohibition_qa: '规章禁止性要求问答', regulation_requirement_qa: '规章要求问答', regulation_responsibility_qa: '规章职责问答', regulation_standard_qa: '规章标准问答', terminology_explanation: '术语解释', terminology_pair: '术语中英文对照', terminology_translation: '术语翻译', textbook_definition_qa: '教材定义问答', textbook_extractive_qa: '教材抽取式问答', textbook_judgment: '教材判断题', textbook_multiple_choice: '教材选择题', textbook_operation_qa: '教材运行检修问答', textbook_source: '教材原文页', textbook_qa: '教材问答', grounded_qa: '证据支撑问答', image_description: '图片描述', multiple_choice: '选择题', judgment: '判断题',
  },
  en: {
    concept_explanation_qa: 'Concept explanation Q&A', regulation_clause_qa: 'Regulation clause Q&A', regulation_definition_qa: 'Regulation definition Q&A', regulation_extractive_qa: 'Regulation extractive Q&A', regulation_inspection_qa: 'Regulation inspection Q&A', regulation_judgment: 'Regulation judgment', regulation_principle_qa: 'Regulation principle Q&A', regulation_prohibition_qa: 'Regulation prohibition Q&A', regulation_requirement_qa: 'Regulation requirement Q&A', regulation_responsibility_qa: 'Regulation responsibility Q&A', regulation_standard_qa: 'Regulation standard Q&A', terminology_explanation: 'Terminology explanation', terminology_pair: 'Terminology bilingual pair', terminology_translation: 'Terminology translation', textbook_definition_qa: 'Textbook definition Q&A', textbook_extractive_qa: 'Textbook extractive Q&A', textbook_judgment: 'Textbook judgment', textbook_multiple_choice: 'Textbook multiple choice', textbook_operation_qa: 'Textbook operation Q&A', textbook_source: 'Textbook source page', textbook_qa: 'Textbook Q&A', grounded_qa: 'Evidence-grounded Q&A', image_description: 'Image description', multiple_choice: 'Multiple choice', judgment: 'Judgment',
  },
  ms: {
    concept_explanation_qa: 'Soal jawab penerangan konsep', regulation_clause_qa: 'Soal jawab klausa peraturan', regulation_definition_qa: 'Soal jawab definisi peraturan', regulation_extractive_qa: 'Soal jawab ekstraktif peraturan', regulation_inspection_qa: 'Soal jawab pemeriksaan peraturan', regulation_judgment: 'Penghakiman peraturan', regulation_principle_qa: 'Soal jawab prinsip peraturan', regulation_prohibition_qa: 'Soal jawab larangan peraturan', regulation_requirement_qa: 'Soal jawab keperluan peraturan', regulation_responsibility_qa: 'Soal jawab tanggungjawab peraturan', regulation_standard_qa: 'Soal jawab piawai peraturan', terminology_explanation: 'Penerangan istilah', terminology_pair: 'Padanan dwibahasa istilah', terminology_translation: 'Terjemahan istilah', textbook_definition_qa: 'Soal jawab definisi buku teks', textbook_extractive_qa: 'Soal jawab ekstraktif buku teks', textbook_judgment: 'Penghakiman buku teks', textbook_multiple_choice: 'Aneka pilihan buku teks', textbook_operation_qa: 'Soal jawab operasi buku teks', textbook_source: 'Halaman sumber buku teks', textbook_qa: 'Soal jawab buku teks', grounded_qa: 'Soal jawab berasaskan bukti', image_description: 'Huraian imej', multiple_choice: 'Aneka pilihan', judgment: 'Penghakiman',
  },
  th: {
    concept_explanation_qa: 'ถามตอบอธิบายแนวคิด', regulation_clause_qa: 'ถามตอบข้อกำหนด', regulation_definition_qa: 'ถามตอบคำนิยามข้อกำหนด', regulation_extractive_qa: 'ถามตอบแบบสกัดจากข้อกำหนด', regulation_inspection_qa: 'ถามตอบการตรวจสอบข้อกำหนด', regulation_judgment: 'คำถามตัดสินข้อกำหนด', regulation_principle_qa: 'ถามตอบหลักการข้อกำหนด', regulation_prohibition_qa: 'ถามตอบข้อห้าม', regulation_requirement_qa: 'ถามตอบข้อกำหนดความต้องการ', regulation_responsibility_qa: 'ถามตอบหน้าที่รับผิดชอบ', regulation_standard_qa: 'ถามตอบมาตรฐานข้อกำหนด', terminology_explanation: 'อธิบายศัพท์', terminology_pair: 'คู่ศัพท์สองภาษา', terminology_translation: 'แปลศัพท์', textbook_definition_qa: 'ถามตอบคำนิยามจากตำรา', textbook_extractive_qa: 'ถามตอบแบบสกัดจากตำรา', textbook_judgment: 'คำถามตัดสินจากตำรา', textbook_multiple_choice: 'ปรนัยจากตำรา', textbook_operation_qa: 'ถามตอบการปฏิบัติงานจากตำรา', textbook_source: 'หน้าต้นฉบับตำรา', textbook_qa: 'ถามตอบตำรา', grounded_qa: 'ถามตอบอ้างอิงหลักฐาน', image_description: 'คำบรรยายภาพ', multiple_choice: 'ปรนัย', judgment: 'ตัดสิน',
  },
  vi: {
    concept_explanation_qa: 'Hỏi đáp giải thích khái niệm', regulation_clause_qa: 'Hỏi đáp điều khoản quy định', regulation_definition_qa: 'Hỏi đáp định nghĩa quy định', regulation_extractive_qa: 'Hỏi đáp trích xuất quy định', regulation_inspection_qa: 'Hỏi đáp kiểm tra quy định', regulation_judgment: 'Câu hỏi đúng sai quy định', regulation_principle_qa: 'Hỏi đáp nguyên tắc quy định', regulation_prohibition_qa: 'Hỏi đáp yêu cầu cấm', regulation_requirement_qa: 'Hỏi đáp yêu cầu quy định', regulation_responsibility_qa: 'Hỏi đáp trách nhiệm quy định', regulation_standard_qa: 'Hỏi đáp tiêu chuẩn quy định', terminology_explanation: 'Giải thích thuật ngữ', terminology_pair: 'Cặp thuật ngữ song ngữ', terminology_translation: 'Dịch thuật ngữ', textbook_definition_qa: 'Hỏi đáp định nghĩa giáo trình', textbook_extractive_qa: 'Hỏi đáp trích xuất giáo trình', textbook_judgment: 'Câu hỏi đúng sai giáo trình', textbook_multiple_choice: 'Trắc nghiệm giáo trình', textbook_operation_qa: 'Hỏi đáp vận hành bảo trì giáo trình', textbook_source: 'Trang nguồn giáo trình', textbook_qa: 'Hỏi đáp giáo trình', grounded_qa: 'Hỏi đáp dựa trên bằng chứng', image_description: 'Mô tả hình ảnh', multiple_choice: 'Trắc nghiệm', judgment: 'Đúng sai',
  },
  kk: {
    concept_explanation_qa: 'Ұғымды түсіндіру сұрақ-жауап', regulation_clause_qa: 'Ереже тармағы сұрақ-жауап', regulation_definition_qa: 'Ереже анықтамасы сұрақ-жауап', regulation_extractive_qa: 'Ережеден үзінді сұрақ-жауап', regulation_inspection_qa: 'Ереже тексеру сұрақ-жауап', regulation_judgment: 'Ереже пайымдау сұрағы', regulation_principle_qa: 'Ереже қағидасы сұрақ-жауап', regulation_prohibition_qa: 'Ереже тыйымы сұрақ-жауап', regulation_requirement_qa: 'Ереже талабы сұрақ-жауап', regulation_responsibility_qa: 'Ереже жауапкершілігі сұрақ-жауап', regulation_standard_qa: 'Ереже стандарты сұрақ-жауап', terminology_explanation: 'Термин түсіндірмесі', terminology_pair: 'Екітілді термин жұбы', terminology_translation: 'Термин аудармасы', textbook_definition_qa: 'Оқулық анықтамасы сұрақ-жауап', textbook_extractive_qa: 'Оқулықтан үзінді сұрақ-жауап', textbook_judgment: 'Оқулық пайымдау сұрағы', textbook_multiple_choice: 'Оқулық көп таңдаулы сұрақ', textbook_operation_qa: 'Оқулық пайдалану-жөндеу сұрақ-жауап', textbook_source: 'Оқулық бастапқы беті', textbook_qa: 'Оқулық сұрақ-жауап', grounded_qa: 'Дәлелге негізделген сұрақ-жауап', image_description: 'Сурет сипаттамасы', multiple_choice: 'Көп таңдаулы', judgment: 'Пайымдау',
  },
}

const statusLabelMap = {
  zh: { pending: '待审核', needs_revision: '需修改', approved: '已通过', rejected: '已驳回', deleted: '已删除' },
  en: { pending: 'Pending', needs_revision: 'Needs revision', approved: 'Approved', rejected: 'Rejected', deleted: 'Deleted' },
  ms: { pending: 'Menunggu', needs_revision: 'Perlu semakan', approved: 'Diluluskan', rejected: 'Ditolak', deleted: 'Dipadam' },
  th: { pending: 'รอตรวจ', needs_revision: 'ต้องแก้ไข', approved: 'อนุมัติแล้ว', rejected: 'ปฏิเสธแล้ว', deleted: 'ลบแล้ว' },
  vi: { pending: 'Chờ duyệt', needs_revision: 'Cần sửa', approved: 'Đã duyệt', rejected: 'Đã từ chối', deleted: 'Đã xóa' },
  kk: { pending: 'Күтуде', needs_revision: 'Түзету керек', approved: 'Бекітілді', rejected: 'Қабылданбады', deleted: 'Жойылды' },
}

const qualityFlagLabelMap = {
  zh: { question_underspecified: '问题不明确', answer_incomplete: '答案不完整', evidence_mismatch: '证据不匹配', ocr_error: 'OCR 识别错误', duplicate: '重复数据', category_error: '分类错误', unsafe_or_uncertain: '不安全或不确定', human_review_required: '需要人工复核' },
  en: { question_underspecified: 'Question unclear', answer_incomplete: 'Answer incomplete', evidence_mismatch: 'Evidence mismatch', ocr_error: 'OCR error', duplicate: 'Duplicate', category_error: 'Category error', unsafe_or_uncertain: 'Unsafe or uncertain', human_review_required: 'Human review required' },
  ms: { question_underspecified: 'Soalan tidak jelas', answer_incomplete: 'Jawapan tidak lengkap', evidence_mismatch: 'Bukti tidak sepadan', ocr_error: 'Ralat OCR', duplicate: 'Pendua', category_error: 'Ralat kategori', unsafe_or_uncertain: 'Tidak selamat atau tidak pasti', human_review_required: 'Perlu semakan manusia' },
  th: { question_underspecified: 'คำถามไม่ชัดเจน', answer_incomplete: 'คำตอบไม่ครบ', evidence_mismatch: 'หลักฐานไม่ตรง', ocr_error: 'ข้อผิดพลาด OCR', duplicate: 'ซ้ำ', category_error: 'หมวดหมู่ผิด', unsafe_or_uncertain: 'ไม่ปลอดภัยหรือไม่แน่ชัด', human_review_required: 'ต้องตรวจโดยมนุษย์' },
  vi: { question_underspecified: 'Câu hỏi chưa rõ', answer_incomplete: 'Câu trả lời chưa đầy đủ', evidence_mismatch: 'Bằng chứng không khớp', ocr_error: 'Lỗi OCR', duplicate: 'Trùng lặp', category_error: 'Sai phân loại', unsafe_or_uncertain: 'Không an toàn hoặc chưa chắc', human_review_required: 'Cần người rà soát' },
  kk: { question_underspecified: 'Сұрақ анық емес', answer_incomplete: 'Жауап толық емес', evidence_mismatch: 'Дәлел сәйкес емес', ocr_error: 'OCR қатесі', duplicate: 'Қайталанған', category_error: 'Санат қатесі', unsafe_or_uncertain: 'Қауіпсіз емес немесе белгісіз', human_review_required: 'Адам тексеруі керек' },
}

const taskTypeLabels = computed(() => taskTypeLabelMap[language.value] || taskTypeLabelMap.zh)
const statusLabels = computed(() => statusLabelMap[language.value] || statusLabelMap.zh)
const qualityFlagLabels = computed(() => qualityFlagLabelMap[language.value] || qualityFlagLabelMap.zh)

function t(key) {
  return uiText[language.value]?.[key] || messageText[language.value]?.[key] || uiText.zh[key] || messageText.zh[key] || key
}

function bilingualLabel(value, labels) {
  const source = labels?.value || labels || {}
  const label = source[value]
  return label ? `${label} / ${value}` : value
}

function qualityFlagLabel(flag) {
  return qualityFlagLabels.value[flag] || flag
}

function labelWithCount(value, labels, counts) {
  return `${bilingualLabel(value, labels)} (${counts?.[value] || 0})`
}

const pageCount = computed(() => Math.max(1, Math.ceil(total.value / filters.page_size)))
const selectedIndex = computed(() => items.value.findIndex((item) => item.id === selected.value?.id))
const selectedBatchCount = computed(() => selectedItemIds.value.size)
const isCurrentPageSelected = computed(() => Boolean(items.value.length) && items.value.every((item) => selectedItemIds.value.has(item.id)))
const statusScopeTotal = computed(() => sumCounts(stats.value.by_status))
const taskScopeTotal = computed(() => sumCounts(stats.value.by_task_type))
const domainScopeTotal = computed(() => sumCounts(stats.value.by_domain_category))
const latestAssistantSources = computed(() => {
  const assistant = [...ragMessages.value].reverse().find((message) => message.role === 'assistant' && message.sources?.length)
  return assistant?.sources || ragResult.value?.sources || []
})

function sumCounts(counts) {
  return Object.values(counts || {}).reduce((sum, value) => sum + Number(value || 0), 0)
}

function resolveImagePath(item) {
  if (!item) return ''
  const metadata = item.metadata_json || {}
  const imagePath = item.source_image_path || metadata.image || metadata.image_path
  if (imagePath) return imagePath
  const ocrPagePath = metadata.ocr_page_path || metadata.markdown
  if (!ocrPagePath || !ocrPagePath.includes('/pages/') || !ocrPagePath.endsWith('.md')) return ''
  return ocrPagePath.replace('/pages/', '/images/').replace(/\.md$/, '.png')
}

const selectedImageUrl = computed(() => {
  const imagePath = resolveImagePath(selected.value)
  return imagePath ? `${apiBase}/api/files?path=${encodeURIComponent(imagePath)}` : ''
})

function fileUrl(path) {
  return `${apiBase}/api/files?path=${encodeURIComponent(path)}`
}

function sourceMarkdownPath(item) {
  const metadata = item?.metadata_json || {}
  return metadata.ocr_page_path || metadata.markdown || ''
}

function resolveMarkdownAssetPath(src, item) {
  if (!src || /^(https?:|data:|blob:)/i.test(src)) return src
  const markdownPath = sourceMarkdownPath(item)
  const params = new URLSearchParams({ src })
  if (markdownPath) params.set('markdown_path', markdownPath)
  if (item?.page_number) params.set('page', String(item.page_number))
  return `${apiBase}/api/ocr-assets?${params.toString()}`
}

function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function normalizeText(value) {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function escapeRegExp(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function splitSourceLines(value) {
  return String(value || '')
    .replace(/\r\n?/g, '\n')
    .split('\n')
}

function extractSourceLineNumber(line) {
  const match = String(line || '').match(/^\s*L(\d+)\s*\|/)
  return match ? Number(match[1]) : null
}

function parseOcrSourceLine(line) {
  const raw = String(line || '')
  const match = raw.match(/^\s*(L\d+)\s*\|\s*(第\s*\d+\s*页)\s*\|\s*(.*)$/)
  if (!match) {
    return {
      lineLabel: '',
      pageLabel: '',
      text: raw,
    }
  }
  return {
    lineLabel: match[1],
    pageLabel: match[2].replace(/\s+/g, ' '),
    text: match[3] || '',
  }
}

function findSourceMatchLine(lines, candidates) {
  const normalizedLines = lines.map((line) => normalizeText(line))
  for (const candidate of candidates) {
    const normalized = normalizeText(candidate)
    if (!normalized) continue
    const fragments = [
      normalized,
      normalized.slice(0, 80),
      normalized.slice(0, 40),
    ].filter((fragment) => fragment.length >= 8)

    for (const fragment of fragments) {
      const index = normalizedLines.findIndex((line) => line.includes(fragment))
      if (index >= 0) return index
    }
  }
  return normalizedLines.findIndex((line) => line.length > 0)
}

function ocrLineNumber(item) {
  const metadata = item?.metadata_json || {}
  const raw = metadata.line_number ?? metadata.ocr_line_number ?? metadata.line
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function findOcrLineIndex(lines, lineNumber) {
  if (!lineNumber) return -1
  const pattern = new RegExp(`^\\s*L${lineNumber}\\s*\\|`)
  return lines.findIndex((line) => pattern.test(String(line || '')))
}

function highlightSourceHtml(text, candidates, lineNumber) {
  const lines = splitSourceLines(text)
  if (lineNumber) {
    const pattern = new RegExp(`^\\s*L${lineNumber}\\s*\\|`)
    return lines
      .map((line) => {
        const escaped = escapeHtml(line || ' ')
        return pattern.test(line) ? `<div class="source-full-line matched">${escaped}</div>` : `<div class="source-full-line">${escaped}</div>`
      })
      .join('')
  }

  let html = escapeHtml(text)
  const terms = candidates
    .map((candidate) => normalizeText(candidate))
    .filter((candidate) => candidate.length >= 8)
    .sort((a, b) => b.length - a.length)
    .slice(0, 3)

  terms.forEach((term) => {
    const pattern = escapeRegExp(term).replace(/\s+/g, '\\s+')
    html = html.replace(new RegExp(pattern, 'g'), '<mark>$&</mark>')
  })
  return html
}

function renderMarkdownToHtml(markdown, item) {
  const source = markdown || ''
  if (!source.trim()) return `<p class="empty-markdown">${escapeHtml(t('emptyMarkdown'))}</p>`

  return source
    .split(/\n{2,}/)
    .map((block) => {
      const trimmed = block.trim()
      if (!trimmed) return ''

      const imageOnly = trimmed.match(/^!\[([^\]]*)\]\(([^)]+)\)$/)
      if (imageOnly) {
    const alt = escapeHtml(imageOnly[1] || t('ocrImage'))
        const src = resolveMarkdownAssetPath(imageOnly[2].trim(), item)
        return `<figure class="markdown-image"><img src="${src}" alt="${alt}" /><figcaption>${alt}</figcaption></figure>`
      }

      const heading = trimmed.match(/^(#{1,4})\s+(.+)$/)
      if (heading) {
        const level = heading[1].length
        return `<h${level}>${escapeHtml(heading[2])}</h${level}>`
      }

      const html = escapeHtml(trimmed).replace(
        /!\[([^\]]*)\]\(([^)]+)\)/g,
        (_match, alt, src) => {
          const imageSrc = resolveMarkdownAssetPath(src.trim(), item)
        return `<figure class="markdown-image inline"><img src="${imageSrc}" alt="${escapeHtml(alt || t('ocrImage'))}" /><figcaption>${escapeHtml(alt || t('ocrImage'))}</figcaption></figure>`
        }
      )
      return `<p>${html.replace(/\n/g, '<br>')}</p>`
    })
    .join('')
}

const renderedSourceHtml = computed(() => {
  const sourceText = editor.source_text || editor.evidence || ''
  const lineNumber = ocrLineNumber(selected.value)
  if (lineNumber && /^\s*L\d+\s*\|/m.test(sourceText)) {
    return highlightSourceHtml(sourceText, [], lineNumber)
  }
  return renderMarkdownToHtml(sourceText, selected.value)
})
const markdownHasImage = computed(() => /!\[[^\]]*\]\([^)]+\)/.test(editor.source_text || editor.evidence || ''))
const sourcePreview = computed(() => {
  const sourceText = editor.source_text || editor.evidence || ''
  const lines = splitSourceLines(sourceText)
  const explicitLineNumber = ocrLineNumber(selected.value)
  const candidates = [
    selected.value?.original_answer,
    editor.answer,
    selected.value?.original_question,
    editor.question,
    editor.evidence,
  ].filter(Boolean)
  const explicitLineIndex = findOcrLineIndex(lines, explicitLineNumber)
  const matchLine = explicitLineIndex >= 0 ? explicitLineIndex : findSourceMatchLine(lines, candidates)
  const contextStart = Math.max(0, matchLine - 20)
  const contextEnd = Math.min(lines.length, matchLine + 21)
  const contextLines = lines.slice(contextStart, contextEnd).map((line, index) => ({
    number: extractSourceLineNumber(line) || contextStart + index + 1,
    ...parseOcrSourceLine(line),
    matched: contextStart + index === matchLine,
  }))
  const fullLines = lines.map((line, index) => ({
    number: extractSourceLineNumber(line) || index + 1,
    ...parseOcrSourceLine(line),
    matched: index === matchLine,
  }))

  return {
    hasText: Boolean(sourceText.trim()),
    contextLines,
    contextPlainText: contextLines.map((line) => line.text).join('\n'),
    fullLines,
    fullPlainText: fullLines.map((line) => line.text).join('\n'),
    fullHtml: highlightSourceHtml(sourceText, candidates, explicitLineNumber),
    matchLine: explicitLineNumber || matchLine + 1,
    matchedByLineNumber: explicitLineIndex >= 0,
  }
})
const evidenceSourceCards = computed(() => {
  const sources = selected.value?.metadata_json?.evidence_sources || {}
  return [
    { key: 'codex', label: `Codex ${t('evidence')}`, payload: sources.codex || {} },
    { key: 'deepseek', label: `DeepSeek ${t('evidence')}`, payload: sources.deepseek || {} },
  ].filter((card) => card.payload.evidence || card.payload.error)
})

function queryString() {
  const params = new URLSearchParams()
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== '' && value !== null) params.set(key, String(value))
  })
  return params.toString()
}

function statsQueryString() {
  const params = new URLSearchParams()
  for (const key of ['status', 'task_type', 'domain_category', 'document_id', 'search']) {
    const value = filters[key]
    if (value !== '' && value !== null) params.set(key, String(value))
  }
  return params.toString()
}

async function api(path, options = {}) {
  let response
  try {
    response = await fetch(`${apiBase}${path}`, {
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      ...options,
    })
  } catch (error) {
    showNetworkDialog(error, path)
    throw new Error('后端服务连接失败')
  }
  if (isGatewayError(response.status)) {
    showNetworkDialog(new Error(`HTTP ${response.status}`), path)
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `${t('requestFailed')}：${response.status}`)
  }
  return response
}

async function loadItems(keepSelection = false) {
  loading.value = true
  try {
    const response = await api(`/api/items?${queryString()}`)
    const data = await response.json()
    items.value = data.items
    total.value = data.total
    const currentIds = new Set(items.value.map((item) => item.id))
    selectedItemIds.value = new Set([...selectedItemIds.value].filter((id) => currentIds.has(id)))
    if (!keepSelection || !items.value.some((item) => item.id === selected.value?.id)) {
      selectItem(items.value[0] || null)
    }
  } catch (error) {
    notice.value = error.message
  } finally {
    loading.value = false
  }
}

async function loadStats() {
  const response = await api(`/api/stats?${statsQueryString()}`)
  stats.value = await response.json()
}

async function loadMeta() {
  const [optionsResponse, documentsResponse] = await Promise.all([
    api('/api/options'),
    api('/api/documents'),
  ])
  options.value = await optionsResponse.json()
  documents.value = await documentsResponse.json()
  await loadStats()
}

function selectItem(item) {
  selected.value = item
  imageZoom.value = 100
  if (!item) return
  Object.assign(editor, {
    task_type: item.task_type || '',
    domain_category: item.domain_category || '',
    knowledge_category: item.knowledge_category || '',
    question: item.question || '',
    answer: item.answer || '',
    question_en: item.question_en || '',
    answer_en: item.answer_en || '',
    evidence: item.evidence || '',
    source_text: item.source_text || '',
    chapter: item.chapter || '',
    page_number: item.page_number,
    quality_flags: [...(item.quality_flags || [])],
    reviewer: editor.reviewer || item.reviewer || '',
    review_comment: item.review_comment || '',
  })
}

function toggleBatchSelection(itemId, checked) {
  const next = new Set(selectedItemIds.value)
  if (checked) next.add(itemId)
  else next.delete(itemId)
  selectedItemIds.value = next
}

function toggleLoadSelection(item, checked) {
  toggleBatchSelection(item.id, checked)
  if (checked) selectItem(item)
}

function toggleCurrentPageSelection(checked) {
  selectedItemIds.value = checked ? new Set(items.value.map((item) => item.id)) : new Set()
}

async function batchReview(status) {
  const itemIds = [...selectedItemIds.value]
  if (!itemIds.length) return
  if (!editor.reviewer.trim() && status !== 'deleted') {
    notice.value = t('reviewerRequired')
    return
  }
  if (status === 'deleted' && !window.confirm(t('confirmDelete'))) return
  saving.value = true
  try {
    localStorage.setItem('railway-reviewer', editor.reviewer)
    const response = await api('/api/items/batch/review', {
      method: 'POST',
      body: JSON.stringify({
        item_ids: itemIds,
        status,
        reviewer: editor.reviewer,
        comment: editor.review_comment,
      }),
    })
    const result = await response.json()
    selectedItemIds.value = new Set()
    notice.value = `${result.updated} ${t('records')}`
    await Promise.all([loadItems(), loadMeta()])
  } catch (error) {
    notice.value = error.message
  } finally {
    saving.value = false
  }
}

async function saveItem() {
  if (!selected.value) return false
  saving.value = true
  try {
    localStorage.setItem('railway-reviewer', editor.reviewer)
    const response = await api(`/api/items/${selected.value.id}`, {
      method: 'PATCH',
      body: JSON.stringify(editor),
    })
    selected.value = await response.json()
    const index = items.value.findIndex((item) => item.id === selected.value.id)
    if (index >= 0) items.value[index] = selected.value
    notice.value = t('saveSuccess')
    return true
  } catch (error) {
    notice.value = error.message
    return false
  } finally {
    saving.value = false
  }
}

async function createNewItem() {
  const base = selected.value || {}
  const metadata = { ...(base.metadata_json || {}) }
  if (base.id) metadata.created_from_item_id = base.id
  const payload = {
    task_type: editor.task_type || base.task_type || 'grounded_qa',
    domain_category: editor.domain_category || base.domain_category || '',
    knowledge_category: editor.knowledge_category || base.knowledge_category || '',
    question: '',
    answer: '',
    question_en: '',
    answer_en: '',
    evidence: editor.evidence || base.evidence || '',
    source_text: editor.source_text || base.source_text || '',
    source_type: base.source_type || 'manual',
    source_document: base.source_document || 'manual',
    source_path: base.source_path || 'manual',
    chapter: editor.chapter || base.chapter || '',
    page_number: editor.page_number ?? base.page_number ?? null,
    quality_flags: [],
    reviewer: editor.reviewer || '',
    review_comment: '',
    metadata_json: metadata,
    document_id: base.document?.id || null,
    original_question: '',
    original_answer: '',
  }
  saving.value = true
  try {
    const response = await api('/api/items', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
    const created = await response.json()
    notice.value = t('createSuccess')
    await Promise.all([loadItems(true), loadMeta()])
    selectItem(created)
  } catch (error) {
    notice.value = error.message
  } finally {
    saving.value = false
  }
}

async function review(status) {
  if (!selected.value) return
  if (!editor.reviewer.trim()) {
    notice.value = t('reviewerRequired')
    return
  }
  if (!(await saveItem())) return
  try {
    await api(`/api/items/${selected.value.id}/review`, {
      method: 'POST',
      body: JSON.stringify({
        status,
        reviewer: editor.reviewer,
        comment: editor.review_comment,
      }),
    })
    notice.value = status === 'approved' ? t('approvedSuccess') : status === 'rejected' ? t('rejectedSuccess') : t('revisionSuccess')
    await Promise.all([loadItems(), loadMeta()])
  } catch (error) {
    notice.value = error.message
  }
}

async function deleteSelected() {
  if (!selected.value) return
  if (!window.confirm(t('confirmDelete'))) return
  try {
    const params = new URLSearchParams()
    if (editor.reviewer.trim()) params.set('reviewer', editor.reviewer.trim())
    if (editor.review_comment.trim()) params.set('comment', editor.review_comment.trim())
    await api(`/api/items/${selected.value.id}?${params.toString()}`, { method: 'DELETE' })
    notice.value = t('deletedSuccess')
    await Promise.all([loadItems(), loadMeta()])
  } catch (error) {
    notice.value = error.message
  }
}

function toggleFlag(flag) {
  const index = editor.quality_flags.indexOf(flag)
  if (index >= 0) editor.quality_flags.splice(index, 1)
  else editor.quality_flags.push(flag)
}

function onTaskTypeChange() {
  if (editor.task_type === 'image_description' && !editor.question.trim()) {
    editor.question = t('imageQuestion')
  }
}

function moveSelection(offset) {
  const next = selectedIndex.value + offset
  if (next >= 0 && next < items.value.length) selectItem(items.value[next])
}

function applyFilters() {
  filters.page = 1
  Promise.all([loadItems(), loadStats()])
}

function applyPageSize(value) {
  const nextSize = Math.max(1, Math.min(500, Number(value) || 30))
  if (filters.page_size === nextSize) return
  filters.page_size = nextSize
  filters.page = 1
  Promise.all([loadItems(), loadStats()])
}

function resetFilters() {
  Object.assign(filters, {
    page: 1,
    page_size: 30,
    status: 'pending',
    task_type: '',
    domain_category: '',
    document_id: '',
    search: '',
  })
  Promise.all([loadItems(), loadStats()])
}

function changePage(delta) {
  const page = filters.page + delta
  if (page < 1 || page > pageCount.value) return
  filters.page = page
  loadItems()
}

async function exportApproved() {
  const response = await api('/api/export?status=approved', { method: 'POST' })
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'railway_corpus_approved_all.jsonl'
  link.click()
  URL.revokeObjectURL(url)
}

async function loadRagStats() {
  try {
    const response = await api('/api/rag/stats')
    ragStats.value = await response.json()
  } catch (error) {
    notice.value = error.message
  }
}

async function loadRagMessages() {
  if (!ragSessionId.value) {
    ragMessages.value = []
    return
  }
  try {
    const response = await api(`/api/rag/sessions/${ragSessionId.value}/messages`)
    ragMessages.value = await response.json()
  } catch (_error) {
    ragSessionId.value = null
    ragMessages.value = []
    localStorage.removeItem('railway-rag-session-id')
  }
}

async function startNewRagSession() {
  const response = await api('/api/rag/sessions', {
    method: 'POST',
    body: JSON.stringify({ title: '新会话' }),
  })
  const session = await response.json()
  ragSessionId.value = session.id
  localStorage.setItem('railway-rag-session-id', String(session.id))
  ragMessages.value = []
  ragResult.value = null
  ragForm.question = ''
}

async function askRag() {
  if (!ragForm.question.trim() || ragLoading.value) return
  const question = ragForm.question.trim()
  ragLoading.value = true
  ragResult.value = null
  try {
    const payload = { ...ragForm, question, session_id: ragSessionId.value }
    const response = await api('/api/rag/ask', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
    ragResult.value = await response.json()
    if (ragResult.value.session_id) {
      ragSessionId.value = ragResult.value.session_id
      localStorage.setItem('railway-rag-session-id', String(ragResult.value.session_id))
    }
    await loadRagMessages()
    if (ragResult.value.audio_url) await playAudioUrl('rag_answer', ragResult.value.audio_url)
    ragForm.question = ''
  } catch (error) {
    notice.value = error.message
  } finally {
    ragLoading.value = false
  }
}

function selectRagExample(question) {
  ragForm.question = question
}

async function loadContextBooks() {
  if (contextBooks.value.length) return
  contextLoading.value = true
  try {
    const response = await api('/api/files?path=data%2Focr%2Frailway_context%2Fmanifest.json')
    const manifest = await response.json()
    contextBooks.value = manifest.books || []
    if (!activeContextBook.value && contextBooks.value.length) {
      await selectContextBook(contextBooks.value[0])
    }
  } catch (error) {
    notice.value = error.message
  } finally {
    contextLoading.value = false
  }
}

async function selectContextBook(book) {
  activeContextBook.value = book
  contextMarkdown.value = ''
  if (!book?.book_context_path) return
  contextLoading.value = true
  try {
    const response = await api(`/api/files?path=${encodeURIComponent(book.book_context_path)}`)
    contextMarkdown.value = await response.text()
  } catch (error) {
    notice.value = error.message
  } finally {
    contextLoading.value = false
  }
}

function switchView(view) {
  stopRecognition()
  activeView.value = view
  if (view === 'rag') {
    if (!ragStats.value.documents) loadRagStats()
    loadRagMessages()
  }
  if (view === 'context') loadContextBooks()
}

function isTypingTarget(target) {
  if (!(target instanceof HTMLElement)) return false
  const tagName = target.tagName.toLowerCase()
  return target.isContentEditable || ['input', 'select', 'textarea'].includes(tagName)
}

function handleReviewShortcut(event) {
  if (activeView.value !== 'review' || !selected.value || saving.value || event.repeat) return
  if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return
  if (isTypingTarget(event.target)) return

  const key = event.key.toLowerCase()
  if (key === 's') {
    event.preventDefault()
    saveItem()
  }
  if (key === 'o') {
    event.preventDefault()
    review('approved')
  }
  if (key === 'p') {
    event.preventDefault()
    review('rejected')
  }
}

watch(notice, (value) => {
  if (value) window.setTimeout(() => (notice.value = ''), 2600)
})

watch(
  () => editor.reviewer,
  (value) => {
    localStorage.setItem('railway-reviewer', value || '')
  },
)

watch(language, (value) => {
  localStorage.setItem('railway-ui-language', value)
})

watch(
  () => selected.value?.id,
  () => stopSpeech(),
)

onMounted(async () => {
  window.addEventListener('keydown', handleReviewShortcut)
  try {
    await Promise.all([loadMeta(), loadItems()])
  } catch (error) {
    notice.value = error.message
  }
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleReviewShortcut)
  stopSpeech()
  stopRecognition()
})
</script>

<template>
  <div class="app-shell" :class="{ 'rag-mode': activeView === 'rag' || activeView === 'context' }">
    <div v-if="showLoadingOverlay" class="loading-overlay" role="status" aria-live="polite">
      <div class="loading-box">
        <RefreshCw class="spin" :size="28" />
      <strong>{{ t('loading') }}</strong>
      </div>
    </div>
    <div v-if="networkDialog.visible" class="network-overlay" role="alertdialog" aria-modal="true">
      <section class="network-dialog">
        <header>
          <div>
            <AlertTriangle :size="22" />
            <strong>{{ networkDialog.title }}</strong>
          </div>
          <button type="button" class="icon-button" @click="hideNetworkDialog" aria-label="关闭">
            <X :size="18" />
          </button>
        </header>
        <p>{{ networkDialog.message }}</p>
        <dl>
          <div>
            <dt>API</dt>
            <dd>{{ networkDialog.endpoint }}</dd>
          </div>
          <div v-if="networkDialog.detail">
            <dt>详情</dt>
            <dd>{{ networkDialog.detail }}</dd>
          </div>
        </dl>
        <footer>
          <button type="button" class="command" @click="hideNetworkDialog">关闭</button>
          <button type="button" class="command primary" @click="checkBackendConnection">
            <RefreshCw :size="16" /> 检测连接
          </button>
        </footer>
      </section>
    </div>
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">RE</div>
        <div>
        <h1>{{ t('appTitle') }}</h1>
        <p>{{ t('appSubtitle') }}</p>
        </div>
      </div>
      <div class="top-actions">
      <label v-if="activeView === 'review'" class="top-reviewer">
        <span>{{ t('reviewer') }}</span>
        <input v-model="editor.reviewer" :placeholder="t('reviewerPlaceholder')" />
      </label>
      <label class="top-language">
        <span>{{ t('language') }}</span>
        <select v-model="language">
          <option v-for="option in languageOptions" :key="option.value" :value="option.value">
            {{ option.label }}
          </option>
        </select>
      </label>
      <div class="view-switch">
        <button :class="{ active: activeView === 'review' }" @click="switchView('review')">
          <BookOpen :size="16" /> {{ t('review') }}
        </button>
        <button :class="{ active: activeView === 'rag' }" @click="switchView('rag')">
          <Bot :size="16" /> {{ t('rag') }}
        </button>
        <button :class="{ active: activeView === 'context' }" @click="switchView('context')">
          <FileText :size="16" /> {{ t('context') }}
        </button>
      </div>
        <button
          v-if="activeView === 'review'"
          class="icon-button"
        :title="t('refresh')"
          @click="Promise.all([loadMeta(), loadItems(true)])"
        >
          <RefreshCw :size="18" />
        </button>
      <button v-if="activeView === 'review'" class="command secondary" :disabled="saving" @click="createNewItem">
        {{ t('create') }}
      </button>
      <button v-if="activeView === 'review'" class="command secondary" @click="exportApproved">
        <Download :size="17" /> {{ t('exportApproved') }}
      </button>
      </div>
    </header>

    <section v-if="activeView === 'review'" class="stats-band">
    <div><span>{{ t('total') }}</span><strong>{{ stats.total }}</strong></div>
    <div><span>{{ t('pending') }}</span><strong>{{ stats.by_status.pending || 0 }}</strong></div>
    <div><span>{{ t('needs_revision') }}</span><strong>{{ stats.by_status.needs_revision || 0 }}</strong></div>
    <div><span>{{ t('approved') }}</span><strong>{{ stats.by_status.approved || 0 }}</strong></div>
    <div><span>{{ t('rejected') }}</span><strong>{{ stats.by_status.rejected || 0 }}</strong></div>
    </section>

    <main v-if="activeView === 'review'" class="workspace">
      <aside class="queue-panel">
        <div class="panel-heading">
          <div><Filter :size="17" /><strong>{{ t('candidateQueue') }}</strong></div>
          <button class="text-button" @click="resetFilters">{{ t('reset') }}</button>
        </div>

        <div class="filters">
          <label class="search-field">
            <Search :size="16" />
            <input v-model="filters.search" :placeholder="t('searchPlaceholder')" @keyup.enter="applyFilters" />
          </label>
          <div class="filter-grid">
            <select v-model="filters.status" @change="applyFilters">
              <option value="">{{ t('allStatus') }} ({{ statusScopeTotal || 0 }})</option>
              <option v-for="status in options.statuses" :key="status" :value="status">
                {{ labelWithCount(status, statusLabels, stats.by_status) }}
              </option>
            </select>
            <select v-model="filters.task_type" @change="applyFilters">
              <option value="">{{ t('allTasks') }} ({{ taskScopeTotal || 0 }})</option>
              <option v-for="type in options.task_types" :key="type" :value="type">
                {{ labelWithCount(type, taskTypeLabels, stats.by_task_type) }}
              </option>
            </select>
            <select v-model="filters.domain_category" @change="applyFilters">
              <option value="">{{ t('allDomains') }} ({{ domainScopeTotal || 0 }})</option>
              <option v-for="category in options.domain_categories" :key="category" :value="category">
                {{ labelWithCount(category, domainLabels, stats.by_domain_category) }}
              </option>
            </select>
            <select v-model="filters.document_id" @change="applyFilters">
              <option value="">{{ t('allDocuments') }}</option>
              <option v-for="document in documents" :key="document.id" :value="document.id">
                {{ document.title }}
              </option>
            </select>
          </div>
        </div>

        <div class="queue-meta">
          <span>{{ total }} {{ t('records') }}</span>
          <span>{{ t('page') }} {{ filters.page }} / {{ pageCount }}</span>
        </div>
        <div class="batch-toolbar">
          <label class="batch-select-all">
            <input
              type="checkbox"
              :checked="isCurrentPageSelected"
              :disabled="!items.length || loading"
              @change="toggleCurrentPageSelection($event.target.checked)"
            />
            <span>全选</span>
          </label>
          <span>{{ selectedBatchCount }} 已选</span>
          <button class="text-button" :disabled="!selectedBatchCount || saving" @click="batchReview('approved')">{{ t('pass') }}</button>
          <button class="text-button danger-text" :disabled="!selectedBatchCount || saving" @click="batchReview('rejected')">{{ t('rejected') }}</button>
          <button class="text-button" :disabled="!selectedBatchCount || saving" @click="batchReview('needs_revision')">{{ t('needs_revision') }}</button>
          <button class="text-button danger-text" :disabled="!selectedBatchCount || saving" @click="batchReview('deleted')">{{ t('delete') }}</button>
          <button class="text-button" :disabled="!selectedBatchCount || saving" @click="toggleCurrentPageSelection(false)">清空</button>
        </div>
        <div class="item-list" :class="{ muted: loading }">
<article
v-for="item in items"
:key="item.id"
class="queue-item"
:class="{ active: selected?.id === item.id, checked: selectedItemIds.has(item.id) }"
>
<div
class="queue-item-main"
role="button"
tabindex="0"
@click="selectItem(item)"
@keydown.enter.prevent="selectItem(item)"
@keydown.space.prevent="selectItem(item)"
>
<div class="item-topline">
<input
class="queue-check"
type="checkbox"
:checked="selectedItemIds.has(item.id)"
                @click.stop
                @change="toggleBatchSelection(item.id, $event.target.checked)"
              />
              <span class="id-label">#{{ item.id }}</span>
              <span class="type-label">{{ item.task_type }}</span>
<span class="status-dot" :data-status="item.review_status"></span>
</div>
<strong>{{ item.question || `${t('page')} ${item.page_number || '-'} OCR` }}</strong>
<span>{{ item.domain_category || t('unclassified') }} · {{ item.source_document }}</span>
</div>
<input
class="queue-load-check"
type="checkbox"
:checked="selectedItemIds.has(item.id)"
:title="t('selectItem')"
:aria-label="t('selectItem')"
@click.stop
@change="toggleLoadSelection(item, $event.target.checked)"
/>
</article>
          <div v-if="!items.length && !loading" class="empty-state">{{ t('noData') }}</div>
        </div>
        <div class="pagination">
          <button class="icon-button" :disabled="filters.page <= 1" :title="t('previousPage')" @click="changePage(-1)">
            <ChevronLeft :size="18" />
          </button>
          <span>{{ filters.page }} / {{ pageCount }}</span>
          <label class="page-size-control">
            <span>{{ t('pageSize') }}</span>
            <select :value="filters.page_size" @change="applyPageSize($event.target.value)">
              <option v-for="size in pageSizeOptions" :key="size" :value="size">{{ size }}</option>
            </select>
            <input
              :value="filters.page_size"
              type="number"
              min="1"
              max="500"
              step="1"
              @change="applyPageSize($event.target.value)"
              @keyup.enter="applyPageSize($event.target.value)"
            />
          </label>
          <button class="icon-button" :disabled="filters.page >= pageCount" :title="t('nextPage')" @click="changePage(1)">
            <ChevronRight :size="18" />
          </button>
        </div>
      </aside>

      <aside class="editor-panel">
        <div class="panel-heading">
          <div><strong>{{ t('editorTitle') }}</strong></div>
          <div class="editor-nav">
            <button class="nav-command" :disabled="selectedIndex <= 0" :title="t('previousRecord')" @click="moveSelection(-1)">
              <ChevronLeft :size="18" />
              {{ t('previousRecord') }}
            </button>
            <span v-if="selected" class="version-label">v{{ selected.version }}</span>
            <button class="nav-command" :disabled="selectedIndex < 0 || selectedIndex >= items.length - 1" :title="t('nextRecord')" @click="moveSelection(1)">
              {{ t('nextRecord') }}
              <ChevronRight :size="18" />
            </button>
          </div>
        </div>

        <div v-if="selected" class="editor-scroll">
          

          <div class="form-row three meta-copy-row">
            <label>
                <span>{{ t('taskType') }}</span>
              <input :value="editor.task_type" readonly />
            </label>
            <label>
                <span>{{ t('domainCategory') }}</span>
              <input :value="editor.domain_category" readonly />
            </label>
            <label>
                <span>{{ t('knowledgeCategory') }}</span>
              <input :value="editor.knowledge_category" readonly />
            </label>
          </div>
          <label>
              <span>{{ t('chapter') }}</span>
              <input v-model="editor.chapter" :placeholder="t('chapterPlaceholder')" />
          </label>
          <div class="form-row two qa-row">
            <label>
                <span>{{ t('questionZh') }}</span>
                <textarea v-model="editor.question" rows="6" :placeholder="t('questionPlaceholder')"></textarea>
              <button
                type="button"
                class="speech-button"
                :class="{ active: speakingField === 'question' }"
                :disabled="!editor.question.trim()"
                @click="speakText('question', editor.question, zhVoice)"
              >
                <Square v-if="speakingField === 'question'" :size="15" />
                <Volume2 v-else :size="16" />
                  {{ speakingField === 'question' ? t('stopSpeech') : t('playQuestionZh') }}
              </button>
            </label>
            <label>
                <span>{{ t('questionEn') }}</span>
                <textarea v-model="editor.question_en" rows="6" :placeholder="t('questionEnPlaceholder')"></textarea>
              <button
                type="button"
                class="speech-button"
                :class="{ active: speakingField === 'question_en' }"
                :disabled="!editor.question_en.trim()"
                @click="speakText('question_en', editor.question_en, enVoice)"
              >
                <Square v-if="speakingField === 'question_en'" :size="15" />
                <Volume2 v-else :size="16" />
                  {{ speakingField === 'question_en' ? t('stopSpeech') : t('playQuestionEn') }}
              </button>
            </label>
            <label>
                <span>{{ t('answerZh') }}</span>
                <textarea v-model="editor.answer" rows="6" :placeholder="t('answerPlaceholder')"></textarea>
              <button
                type="button"
                class="speech-button"
                :class="{ active: speakingField === 'answer' }"
                :disabled="!editor.answer.trim()"
                @click="speakText('answer', editor.answer, zhVoice)"
              >
                <Square v-if="speakingField === 'answer'" :size="15" />
                <Volume2 v-else :size="16" />
                  {{ speakingField === 'answer' ? t('stopSpeech') : t('playAnswerZh') }}
              </button>
            </label>
            <label>
                <span>{{ t('answerEn') }}</span>
                <textarea v-model="editor.answer_en" rows="6" :placeholder="t('answerEnPlaceholder')"></textarea>
              <button
                type="button"
                class="speech-button"
                :class="{ active: speakingField === 'answer_en' }"
                :disabled="!editor.answer_en.trim()"
                @click="speakText('answer_en', editor.answer_en, enVoice)"
              >
                <Square v-if="speakingField === 'answer_en'" :size="15" />
                <Volume2 v-else :size="16" />
                  {{ speakingField === 'answer_en' ? t('stopSpeech') : t('playAnswerEn') }}
              </button>
            </label>
          </div>
          <section v-if="evidenceSourceCards.length" class="model-evidence-grid">
            <article v-for="card in evidenceSourceCards" :key="card.key" class="model-evidence-card">
              <header>
                <strong>{{ card.label }}</strong>
                <span v-if="card.payload.model">{{ card.payload.model }}</span>
              </header>
              <p v-if="card.payload.error" class="model-evidence-error">{{ card.payload.error }}</p>
              <pre v-else>{{ card.payload.evidence }}</pre>
              <footer v-if="card.payload.source_label || card.payload.context_path || card.payload.generated_at">
                <span v-if="card.payload.source_label">{{ card.payload.source_label }}</span>
                <span v-else-if="card.payload.context_path">{{ card.payload.context_path }}</span>
                <span v-if="card.payload.generated_at">{{ card.payload.generated_at }}</span>
              </footer>
            </article>
          </section>

          <fieldset class="flags">
            <legend>{{ t('qualityFlags') }}</legend>
            <button
              v-for="flag in options.quality_flags"
              :key="flag"
              type="button"
              :class="{ selected: editor.quality_flags.includes(flag) }"
              @click="toggleFlag(flag)"
            >
              {{ qualityFlagLabel(flag) }}
            </button>
          </fieldset>

          <div class="form-row two">
            <label>
                <span>{{ t('pageNumber') }}</span>
              <input v-model.number="editor.page_number" type="number" min="1" />
            </label>
          </div>
          <label>
              <span>{{ t('reviewComment') }}</span>
            <textarea v-model="editor.review_comment" rows="3"></textarea>
          </label>
          <details v-if="editor.source_text && editor.source_text !== editor.evidence" class="editor-raw-ocr">
            <summary>{{ t('rawOcrMarkdown') }}</summary>
            <textarea class="raw-ocr-textarea" readonly :value="editor.source_text" rows="12"></textarea>
          </details>
        </div>

        <div v-if="selected" class="review-actions">
          <button class="command secondary" :disabled="saving" title="S" @click="saveItem">
            <Save :size="17" /> {{ t('save') }} (S)
          </button>
          <button class="command warning" @click="review('needs_revision')">
            <AlertTriangle :size="17" /> {{ t('needs_revision') }}
          </button>
          <button class="command danger" :disabled="saving" title="P" @click="review('rejected')">
            <X :size="17" /> {{ t('rejected') }} (P)
          </button>
          <button class="command danger ghost" @click="deleteSelected">
            <Trash2 :size="17" /> {{ t('delete') }}
          </button>
          <button class="command primary" :disabled="saving" title="O" @click="review('approved')">
            <Check :size="17" /> {{ t('pass') }} (O)
          </button>
        </div>
      </aside>

      <aside class="ocr-panel">
        <div class="panel-heading">
          <div><FileText :size="17" /><strong>{{ t('ocrPreview') }}</strong></div>
        </div>
        <div v-if="selected" class="ocr-panel-scroll">
          <section class="review-source">
            <dl class="source-meta">
              <div><dt>ID</dt><dd>#{{ selected.id }}</dd></div>
                <div><dt>{{ t('document') }}</dt><dd>{{ selected.source_document || t('unrecorded') }}</dd></div>
                <div><dt>{{ t('pageNumber') }}</dt><dd>{{ selected.page_number || t('unrecorded') }}</dd></div>
                <div><dt>{{ t('chapter') }}</dt><dd>{{ editor.chapter || t('pendingLabel') }}</dd></div>
                <div><dt>{{ t('sourceType') }}</dt><dd>{{ selected.source_type }}</dd></div>
            </dl>
          <figure v-if="selectedImageUrl" class="ocr-image">
            <figcaption class="ocr-image-toolbar">
                  <span>
                    {{ t('ocrImage') }}
                    <template v-if="sourcePreview.hasText"> · {{ t('matchedLine') }} L{{ sourcePreview.matchLine }}</template>
                  </span>
              <label>
                    <span>{{ t('zoom') }}</span>
                <input v-model.number="imageZoom" type="range" min="60" max="220" step="10" />
                <output>{{ imageZoom }}%</output>
              </label>
            </figcaption>
            <div class="ocr-image-viewport">
                  <img :src="selectedImageUrl" :alt="t('ocrImage')" :style="{ transform: `scale(${imageZoom / 100})` }" />
            </div>
          </figure>
            <div class="evidence-block">
                <h2>{{ t('ocrTextPreview') }}</h2>
              <p v-if="sourcePreview.hasText" class="matched-line-note">
                {{ t('matchedLine') }} L{{ sourcePreview.matchLine }}
                <span v-if="sourcePreview.matchedByLineNumber">({{ t('matchedByMetadata') }})</span>
              </p>
              <div v-if="sourcePreview.hasText" class="source-context-copy-grid preview">
                <div class="source-context-meta" aria-hidden="true">
                  <div
                    v-for="line in sourcePreview.fullLines"
                    :key="line.number"
                    class="source-context-meta-row"
                    :class="{ matched: line.matched }"
                  >
                    <span class="source-line-token">{{ line.lineLabel || `L${line.number}` }}</span>
                    <span class="source-page-token">{{ line.pageLabel || '-' }}</span>
                  </div>
                </div>
                <div class="source-context-textbox" role="textbox" aria-readonly="true">
                  <div
                    v-for="line in sourcePreview.fullLines"
                    :key="line.number"
                    class="source-text-row"
                    :class="{ matched: line.matched }"
                  >{{ line.text || ' ' }}</div>
                </div>
              </div>
              <div v-else class="markdown-preview" v-html="renderedSourceHtml"></div>
            </div>
            <div v-if="sourcePreview.hasText" class="source-context-block">
              <div class="source-context-section">
                  <h2>{{ t('sourceContext') }}</h2>
                  <p>{{ t('sourceContextHint') }}</p>
                <div class="source-context-copy-grid">
                  <div class="source-context-meta" aria-hidden="true">
                    <div
                      v-for="line in sourcePreview.contextLines"
                      :key="line.number"
                      class="source-context-meta-row"
                      :class="{ matched: line.matched }"
                    >
                      <span class="source-line-token">{{ line.lineLabel || `L${line.number}` }}</span>
                      <span class="source-page-token">{{ line.pageLabel || '-' }}</span>
                    </div>
                  </div>
                  <div class="source-context-textbox" role="textbox" aria-readonly="true">
                    <div
                      v-for="line in sourcePreview.contextLines"
                      :key="line.number"
                      class="source-text-row"
                      :class="{ matched: line.matched }"
                    >{{ line.text || ' ' }}</div>
                  </div>
                </div>
              </div>
              <div class="source-context-section">
                  <h2>{{ t('fullSource') }}</h2>
                <div class="source-context-copy-grid full">
                  <div class="source-context-meta" aria-hidden="true">
                    <div
                      v-for="line in sourcePreview.fullLines"
                      :key="line.number"
                      class="source-context-meta-row"
                      :class="{ matched: line.matched }"
                    >
                      <span class="source-line-token">{{ line.lineLabel || `L${line.number}` }}</span>
                      <span class="source-page-token">{{ line.pageLabel || '-' }}</span>
                    </div>
                  </div>
                  <div class="source-context-textbox" role="textbox" aria-readonly="true">
                    <div
                      v-for="line in sourcePreview.fullLines"
                      :key="line.number"
                      class="source-text-row"
                      :class="{ matched: line.matched }"
                    >{{ line.text || ' ' }}</div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
          <div v-else class="empty-state">{{ t('selectItem') }}</div>
      </aside>
    </main>

    <main v-else-if="activeView === 'rag'" class="rag-workspace">
      <aside class="rag-control-panel">
        <div class="panel-heading">
          <div><Bot :size="17" /><strong>{{ t('ragTitle') }}</strong></div>
          <button type="button" class="mini-command" @click="startNewRagSession">新会话</button>
        </div>
        <div class="rag-controls">
          <label>
              <span>{{ t('question') }}</span>
            <textarea
              v-model="ragForm.question"
              rows="7"
              :placeholder="t('searchPlaceholder')"
              @keydown.ctrl.enter.prevent="askRag"
            ></textarea>
            <div class="voice-action-row">
              <button
                type="button"
                class="speech-button"
                :class="{ active: recognizingField === 'rag_question' }"
                @click="startSpeechRecognition('rag_question')"
              >
                <MicOff v-if="recognizingField === 'rag_question'" :size="15" />
                <Mic v-else :size="16" />
                {{ recognizingField === 'rag_question' ? t('stopRecording') : t('voiceAsk') }}
              </button>
            </div>
          </label>
          <div class="form-row two">
            <label>
              <span>{{ t('evidenceCount') }}</span>
              <select v-model.number="ragForm.top_k">
                <option :value="3">3</option>
                <option :value="5">5</option>
                <option :value="8">8</option>
              </select>
            </label>
            <label class="toggle-field">
              <span>{{ t('answerMode') }}</span>
              <button
                type="button"
                class="mode-toggle"
                :class="{ active: ragForm.generate }"
                @click="ragForm.generate = !ragForm.generate"
              >
                {{ ragForm.generate ? t('generatingMode') : t('retrievalOnly') }}
              </button>
            </label>
          </div>
          <button class="command primary rag-submit" :disabled="ragLoading" @click="askRag">
            <RefreshCw v-if="ragLoading" class="spin" :size="17" />
            <Send v-else :size="17" />
            {{ ragLoading ? t('generating') : t('submitQuestion') }}
          </button>

          <div class="rag-examples">
            <strong>{{ t('sampleQuestions') }}</strong>
            <button v-for="question in ragExamples" :key="question" @click="selectRagExample(question)">
              {{ question }}
            </button>
          </div>

          <dl class="rag-index-meta">
          <div><dt>会话</dt><dd>{{ ragSessionId || '未创建' }}</dd></div>
          <div><dt>{{ t('model') }}</dt><dd>{{ ragStats.model || t('loading') }}</dd></div>
          <div><dt>{{ t('testIsolation') }}</dt><dd>{{ ragStats.excludes_test_split ? t('enabled') : t('disabled') }}</dd></div>
          </dl>
        </div>
      </aside>

      <section class="rag-result-panel">
        <div class="panel-heading">
          <div><FileText :size="17" /><strong>{{ t('answerAndEvidence') }}</strong></div>
          <span v-if="ragResult" class="rag-status">
            {{ t('retrievalMs') }} {{ ragResult.retrieval_ms }}ms · {{ t('generationMs') }} {{ ragResult.generation_ms }}ms
          </span>
        </div>
        <div v-if="ragLoading" class="rag-loading">
          <RefreshCw class="spin" :size="24" />
            <strong>{{ t('loadingRag') }}</strong>
            <span>{{ t('modelWarmup') }}</span>
        </div>
        <div v-else-if="ragMessages.length || ragResult" class="rag-result-scroll">
          <section class="rag-chat-log">
            <article
              v-for="message in ragMessages"
              :key="message.id"
              class="chat-message"
              :data-role="message.role"
            >
              <header>
                <strong>{{ message.role === 'user' ? '我' : '铁路问答助手' }}</strong>
                <span>{{ new Date(message.created_at).toLocaleString() }}</span>
              </header>
              <p>{{ message.content }}</p>
            </article>
          </section>

          <section v-if="ragResult" class="rag-answer">
            <div class="answer-meta">
              <span>{{ ragResult.mode }}</span>
              <span v-if="ragResult.model">{{ ragResult.model }}</span>
            </div>
            <button
              type="button"
              class="speech-button"
              :class="{ active: speakingField === 'rag_answer' }"
              :disabled="!ragResult.answer"
              @click="ragResult.audio_url ? playAudioUrl('rag_answer', ragResult.audio_url, { toggle: true }) : speakText('rag_answer', ragResult.answer)"
            >
              <Square v-if="speakingField === 'rag_answer'" :size="15" />
              <Volume2 v-else :size="16" />
              {{ speakingField === 'rag_answer' ? t('stopSpeech') : t('playAnswerResult') }}
            </button>
          </section>

          <section v-if="latestAssistantSources.length" class="rag-sources">
            <h2>{{ t('retrievalEvidence') }}</h2>
            <article v-for="(source, index) in latestAssistantSources" :key="source.item_id + '-' + index">
              <header>
                <strong>[{{ t('retrievalEvidence') }} {{ index + 1 }}] {{ source.source_document }}</strong>
                <span>{{ t('relevance') }} {{ source.score }}</span>
              </header>
              <div class="source-tags">
              <span>{{ source.domain_category || t('unclassified') }}</span>
                <span>{{ source.task_type }}</span>
              <span v-if="source.page_number">{{ t('page') }} {{ source.page_number }}</span>
                <span :data-status="source.review_status">{{ source.review_status }}</span>
              </div>
              <p>{{ source.evidence }}</p>
            </article>
          </section>
        </div>
        <div v-else class="empty-state center">
          {{ t('askHint') }}
        </div>
      </section>
    </main>

    <main v-else class="context-workspace">
      <aside class="context-book-panel">
        <div class="panel-heading">
          <div><FileText :size="17" /><strong>{{ t('context') }}</strong></div>
          <span class="rag-status">{{ contextBooks.length }} {{ t('contextBooks') }}</span>
        </div>
        <div class="context-book-list">
          <button
            v-for="book in contextBooks"
            :key="book.book"
            class="queue-item"
            :class="{ active: activeContextBook?.book === book.book }"
            @click="selectContextBook(book)"
          >
            <strong>{{ book.book }}</strong>
              <span>{{ book.pages }} {{ t('pages') }} · {{ book.book_context_path }}</span>
          </button>
        </div>
      </aside>
      <section class="context-doc-panel">
        <div class="panel-heading">
          <div><FileText :size="17" /><strong>{{ activeContextBook?.book || t('chooseBook') }}</strong></div>
          <span class="rag-status">{{ activeContextBook?.pages || 0 }} {{ t('pages') }}</span>
        </div>
        <div v-if="contextLoading" class="rag-loading">
          <RefreshCw class="spin" :size="24" />
          <span>{{ t('loadingContext') }}</span>
        </div>
        <pre v-else class="context-markdown">{{ contextMarkdown }}</pre>
      </section>
    </main>

    <div v-if="notice" class="toast">{{ notice }}</div>
  </div>
</template>
