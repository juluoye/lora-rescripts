import fs from 'fs';
import path from 'path';

const srcRoot = 'H:\\webapp';
const dstRoot = 'H:\\webapp\\github\\lulynx-lora-share';

function read(filePath) {
  return fs.readFileSync(filePath, 'utf8');
}

function write(filePath, content) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content, 'utf8');
}

function copy(relativePath) {
  const src = path.join(srcRoot, relativePath);
  const dst = path.join(dstRoot, relativePath);
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
  console.log(`copied ${relativePath}`);
}

function replaceOnce(content, needle, replacement, label) {
  if (!content.includes(needle)) {
    const trimmedReplacement = replacement.trim();
    if (!trimmedReplacement) {
      return content;
    }
    if (content.includes(trimmedReplacement)) {
      return content;
    }
    throw new Error(`Could not find ${label}`);
  }
  return content.replace(needle, replacement);
}

function insertAfter(content, needle, insertion, label) {
  if (content.includes(insertion.trim())) {
    return content;
  }
  if (!content.includes(needle)) {
    throw new Error(`Could not find ${label}`);
  }
  return content.replace(needle, `${needle}${insertion}`);
}

function insertBefore(content, needle, insertion, label) {
  if (content.includes(insertion.trim())) {
    return content;
  }
  if (!content.includes(needle)) {
    throw new Error(`Could not find ${label}`);
  }
  return content.replace(needle, `${insertion}${needle}`);
}

function updateFrontendApp() {
  const filePath = path.join(dstRoot, 'frontend', 'src', 'App.tsx');
  let content = read(filePath);

  content = insertAfter(
    content,
    "import ArtistThreadForm from './pages/ArtistThreadForm';\n",
    "import TrainingPresetForm from './pages/TrainingPresetForm';\nimport TrainingPresetView from './pages/TrainingPresetView';\n",
    'ArtistThreadForm import',
  );
  content = insertAfter(
    content,
    "import Profile from './pages/Profile';\n",
    "import ExternalKeys from './pages/ExternalKeys';\n",
    'Profile import',
  );

  const newRoutes = `        <Route\n          path="/external-keys"\n          element={\n            <ProtectedRoute>\n              <ExternalKeys />\n            </ProtectedRoute>\n          }\n        />\n        <Route\n          path="/add-training-preset"\n          element={\n            <PermissionRoute permission="can_upload">\n              <TrainingPresetForm />\n            </PermissionRoute>\n          }\n        />\n        <Route\n          path="/training-presets/:id"\n          element={\n            <ProtectedRoute>\n              <TrainingPresetView />\n            </ProtectedRoute>\n          }\n        />\n        <Route\n          path="/edit-training-preset/:id"\n          element={\n            <PermissionRoute permission="can_edit">\n              <TrainingPresetForm />\n            </PermissionRoute>\n          }\n        />\n`;
  content = insertBefore(content, '        <Route path="*" element={<Navigate to="/" replace />} />\n', newRoutes, 'fallback route');

  write(filePath, content);
}

function updateFrontendTypes() {
  const src = path.join(srcRoot, 'frontend', 'src', 'types', 'index.ts');
  const dst = path.join(dstRoot, 'frontend', 'src', 'types', 'index.ts');
  let content = read(src);

  content = content.replace(/^\s*can_use_nai: boolean;\r?\n/gm, '');
  content = content.replace(/access_type:\s*'lora'\s*\|\s*'nai';/g, "access_type: 'lora';");

  write(dst, content);
}

function updateFrontendApi() {
  const src = path.join(srcRoot, 'frontend', 'src', 'utils', 'api.ts');
  const dst = path.join(dstRoot, 'frontend', 'src', 'utils', 'api.ts');
  let content = read(src);
  const marker = '// NovelAI API';
  const idx = content.indexOf(marker);
  if (idx >= 0) {
    content = `${content.slice(0, idx).trimEnd()}\n`;
  }
  write(dst, content);
}

function updateBackendUserModel() {
  const filePath = path.join(dstRoot, 'backend', 'src', 'models', 'user.js');
  let content = read(filePath);

  if (!content.includes('getRawById:')) {
    content = insertAfter(
      content,
      "  getById: (id) => {\n    const stmt = db.prepare('SELECT * FROM users WHERE id = ?');\n    const user = stmt.get(id);\n    return userModel._formatUser(user);\n  },\n",
      "\n  // 根据 ID 获取原始用户记录（包含敏感字段，用于鉴权等内部逻辑）\n  getRawById: (id) => {\n    const stmt = db.prepare('SELECT * FROM users WHERE id = ?');\n    return stmt.get(id) || null;\n  },\n",
      'userModel.getById block',
    );
  }

  if (!content.includes('clearExpiredLock:')) {
    content = insertAfter(
      content,
      "  isAccountLocked: (user) => {\n    if (!user.locked_until) return false;\n    return new Date(user.locked_until) > new Date();\n  },\n",
      "\n  // 清除已过期的锁定状态\n  clearExpiredLock: (user) => {\n    if (!user?.locked_until) return false;\n    const lockedUntil = new Date(user.locked_until);\n    if (Number.isNaN(lockedUntil.getTime()) || lockedUntil > new Date()) {\n      return false;\n    }\n    const stmt = db.prepare('UPDATE users SET failed_login_attempts = 0, locked_until = NULL WHERE id = ?');\n    stmt.run(user.id);\n    return true;\n  },\n",
      'userModel.isAccountLocked block',
    );
  }

  write(filePath, content);
}

function updateBackendMiddleware() {
  const src = path.join(srcRoot, 'backend', 'src', 'middleware', 'auth.js');
  const dst = path.join(dstRoot, 'backend', 'src', 'middleware', 'auth.js');
  write(dst, read(src));
}

function updateBackendAuthRoute() {
  const filePath = path.join(dstRoot, 'backend', 'src', 'routes', 'auth.js');
  let content = read(filePath);

  content = insertAfter(
    content,
    "import { notificationModel } from '../models/notification.js';\n",
    "import { apiKeyModel } from '../models/apiKey.js';\n",
    'notificationModel import',
  );

  content = replaceOnce(
    content,
    "import { authenticate } from '../middleware/auth.js';",
    "import { authenticate, requireScope } from '../middleware/auth.js';",
    'middleware auth import',
  );

  if (!content.includes('ALLOWED_API_KEY_SCOPES')) {
    content = insertAfter(
      content,
      "const LOCK_DURATION_MINUTES = 30;\n",
      "\nconst ALLOWED_API_KEY_SCOPES = new Set(['account:read', 'training_presets:read', 'training_presets:write']);\n\nfunction normalizeApiKeyScopes(input) {\n  const raw = Array.isArray(input) ? input : [];\n  return Array.from(new Set(\n    raw\n      .map(scope => String(scope || '').trim())\n      .filter(scope => ALLOWED_API_KEY_SCOPES.has(scope))\n  ));\n}\n\nfunction ensureJwtSession(req, res, next) {\n  if (req.auth?.type === 'api_key') {\n    return res.status(403).json({ error: '请在普通网页登录态下管理 API Key，不支持使用另一个 API Key 管理。' });\n  }\n  next();\n}\n",
      'LOCK_DURATION_MINUTES block',
    );
  }

  content = replaceOnce(
    content,
    "// 验证 token 接口\nrouter.get('/verify', authenticate, (req, res) => {\n  res.json({ valid: true, user: req.user });\n});\n",
    "// 验证 token 接口\nrouter.get('/verify', authenticate, requireScope('account:read'), (req, res) => {\n  res.json({\n    valid: true,\n    auth_type: req.auth?.type || 'jwt',\n    scopes: req.auth?.scopes || ['*'],\n    user: req.user,\n  });\n});\n\nrouter.get('/api-keys', authenticate, ensureJwtSession, (req, res) => {\n  res.json({ api_keys: apiKeyModel.listByUser(req.user.id) });\n});\n\nrouter.post('/api-keys', authenticate, ensureJwtSession, (req, res) => {\n  const { name, note, scopes, expires_in_days } = req.body || {};\n  const normalizedName = String(name || '').trim();\n  if (!normalizedName) {\n    return res.status(400).json({ error: '请填写 API Key 名称。' });\n  }\n\n  const normalizedScopes = normalizeApiKeyScopes(scopes);\n  if (normalizedScopes.length === 0) {\n    return res.status(400).json({ error: '请至少选择一个权限范围。' });\n  }\n\n  const created = apiKeyModel.create({\n    user_id: req.user.id,\n    name: normalizedName,\n    note: String(note || '').trim(),\n    scopes: normalizedScopes,\n    expires_in_days: expires_in_days === 0 || expires_in_days === '0' ? 0 : Number(expires_in_days || 90),\n  });\n\n  auditLogModel.create({\n    user_id: req.user.id,\n    action: 'create_api_key',\n    target_type: 'api_key',\n    target_id: created.record.id,\n    details: {\n      name: created.record.name,\n      scopes: created.record.scopes,\n      expires_at: created.record.expires_at,\n      key_prefix: created.record.key_prefix,\n    },\n    ip_address: req.ip,\n  });\n\n  res.status(201).json({\n    message: 'API Key 已创建，请立即复制保存。完整 Key 只会展示这一次。',\n    api_key: created.token,\n    record: created.record,\n  });\n});\n\nrouter.delete('/api-keys/:id', authenticate, ensureJwtSession, (req, res) => {\n  const id = Number(req.params.id);\n  if (!Number.isFinite(id) || id <= 0) {\n    return res.status(400).json({ error: 'API Key ID 无效。' });\n  }\n\n  const existing = apiKeyModel.getById(id, req.user.id);\n  if (!existing) {\n    return res.status(404).json({ error: '未找到对应的 API Key。' });\n  }\n\n  const success = apiKeyModel.revoke(id, req.user.id);\n  if (!success) {\n    return res.status(500).json({ error: '撤销 API Key 失败。' });\n  }\n\n  auditLogModel.create({\n    user_id: req.user.id,\n    action: 'revoke_api_key',\n    target_type: 'api_key',\n    target_id: id,\n    details: {\n      name: existing.name,\n      key_prefix: existing.key_prefix,\n    },\n    ip_address: req.ip,\n  });\n\n  res.json({ success: true });\n});\n",
    'verify route block',
  );

  write(filePath, content);
}

function updateBackendServer() {
  const filePath = path.join(dstRoot, 'backend', 'server.js');
  let content = read(filePath);

  content = replaceOnce(content, "import jwt from 'jsonwebtoken';\n", '', 'jwt import');
  content = insertAfter(
    content,
    "import artistThreadRoutes from './src/routes/artistThreads.js';\n",
    "import trainingPresetRoutes from './src/routes/trainingPresets.js';\n",
    'artistThreadRoutes import',
  );
  content = insertAfter(
    content,
    "import { startScheduledBackups } from './src/services/backupService.js';\n",
    "import { hasValidToken } from './src/middleware/auth.js';\n",
    'startScheduledBackups import',
  );

  const legacyHasValidToken = `// 检查是否携带有效 token\nconst hasValidToken = (req) => {\n  const token = req.headers.authorization?.split(' ')[1];\n  if (!token) return false;\n  try {\n    jwt.verify(token, process.env.JWT_SECRET);\n    return true;\n  } catch { return false; }\n};\n\n`;
  content = content.replace(legacyHasValidToken, '');

  content = insertAfter(
    content,
    "app.use('/api/artist-threads', artistThreadRoutes);\n",
    "app.use('/api/training-presets', trainingPresetRoutes);\napp.use('/api/training_presets', trainingPresetRoutes);\n",
    'artist threads route mount',
  );

  write(filePath, content);
}

function updateBackendPackageJson() {
  const filePath = path.join(dstRoot, 'backend', 'package.json');
  const data = JSON.parse(read(filePath));
  data.dependencies = data.dependencies || {};
  data.dependencies.toml = '^4.1.1';

  const orderedDependencies = Object.fromEntries(
    Object.entries(data.dependencies).sort(([a], [b]) => a.localeCompare(b)),
  );
  data.dependencies = orderedDependencies;
  write(filePath, `${JSON.stringify(data, null, 2)}\n`);
}

function updateAccessRequestModel() {
  const filePath = path.join(dstRoot, 'backend', 'src', 'models', 'accessRequest.js');
  let content = read(filePath);

  if (!content.includes("WHEN 'training_preset' THEN tp.title")) {
    content = content.replace(
      /(\s+WHEN 'workflow' THEN w\.name\r?\n)/,
      `$1                  WHEN 'training_preset' THEN tp.title\n`,
    );
  }
  if (!content.includes("WHEN 'training_preset' THEN tp.cover_image")) {
    content = content.replace(
      /(\s+WHEN 'workflow' THEN w\.preview_image\r?\n)/,
      `$1                  WHEN 'training_preset' THEN tp.cover_image\n`,
    );
  }
  if (!content.includes("LEFT JOIN training_presets tp ON ar.item_type = 'training_preset' AND ar.item_id = tp.id")) {
    content = content.replace(
      /(\s+LEFT JOIN workflows w ON ar\.item_type = 'workflow' AND ar\.item_id = w\.id\r?\n)/,
      `$1        LEFT JOIN training_presets tp ON ar.item_type = 'training_preset' AND ar.item_id = tp.id\n`,
    );
  }

  write(filePath, content);
}

function updateAccessRequestRoute() {
  const filePath = path.join(dstRoot, 'backend', 'src', 'routes', 'accessRequests.js');
  let content = read(filePath);

  content = insertAfter(
    content,
    "import { workflowModel } from '../models/workflow.js';\n",
    "import { trainingPresetModel } from '../models/trainingPreset.js';\n",
    'workflowModel import',
  );

  content = replaceOnce(
    content,
    "    case 'workflow': return workflowModel;\n    default: return null;\n",
    "    case 'workflow': return workflowModel;\n    case 'training_preset': return trainingPresetModel;\n    default: return null;\n",
    'getModel switch',
  );

  content = content.replaceAll("['lora', 'prompt', 'workflow']", "['lora', 'prompt', 'workflow', 'training_preset']");
  content = content.replaceAll("auditLogModel.add(", 'auditLogModel.create(');
  content = content.replaceAll("{ lora: 'Lora', prompt: '提示词', workflow: '工作流' }", "{ lora: 'Lora', prompt: '提示词', workflow: '工作流', training_preset: '炼丹参数' }");
  content = content.replaceAll("`/?tab=${itemType}s`", "itemType === 'training_preset' ? '/home?tab=training_presets' : `/?tab=${itemType}s`");
  content = content.replaceAll("`/?tab=${request.item_type}s`", "request.item_type === 'training_preset' ? '/home?tab=training_presets' : `/?tab=${request.item_type}s`");

  if (!content.includes('const myTrainingPresets = trainingPresetModel.getByUploader(userId);')) {
    content = replaceOnce(
      content,
      "    const myLoras = loraModel.getByUploader(userId);\n    const myPrompts = promptModel.getByUploader(userId);\n    const myWorkflows = workflowModel.getByUploader(userId);\n",
      "    const myLoras = loraModel.getByUploader(userId);\n    const myPrompts = promptModel.getByUploader(userId);\n    const myWorkflows = workflowModel.getByUploader(userId);\n    const myTrainingPresets = trainingPresetModel.getByUploader(userId);\n",
      'my-items source lists',
    );

    content = replaceOnce(
      content,
      "    // 按时间排序\n    allRequests.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));\n",
      "    // 获取每个训练参数的申请\n    for (const item of myTrainingPresets) {\n      if (item.access_type === 'request') {\n        const requests = accessRequestModel.getRequestsByItem('training_preset', item.id, status === 'all' ? null : status);\n        requests.forEach(r => allRequests.push({ ...r, item_type: 'training_preset', item_name: item.name, item_image: item.preview_image }));\n      }\n    }\n\n    // 按时间排序\n    allRequests.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));\n",
      'my-items training preset loop',
    );
  }

  content = content.replace(
    /(const myWorkflows = workflowModel\.getByUploader\(userId\);\r?\n)(\s*const myTrainingPresets = trainingPresetModel\.getByUploader\(userId\);\r?\n)+/g,
    "$1    const myTrainingPresets = trainingPresetModel.getByUploader(userId);\n",
  );

  content = content.replace(
    /(\/\/ 获取用户上传的所有项目\r?\n\s*const myLoras = loraModel\.getByUploader\(userId\);\r?\n\s*const myPrompts = promptModel\.getByUploader\(userId\);\r?\n\s*const myWorkflows = workflowModel\.getByUploader\(userId\);\r?\n)(\s*let pendingCount = 0;)/,
    "$1    const myTrainingPresets = trainingPresetModel.getByUploader(userId);\n\n    $2",
  );

  if (!content.includes("const requests = accessRequestModel.getRequestsByItem('training_preset', item.id, 'pending');")) {
    content = replaceOnce(
      content,
      "    for (const item of myWorkflows) {\n      if (item.access_type === 'request') {\n        const requests = accessRequestModel.getRequestsByItem('workflow', item.id, 'pending');\n        pendingCount += requests.length;\n      }\n    }\n\n    res.json({ count: pendingCount });\n",
      "    for (const item of myWorkflows) {\n      if (item.access_type === 'request') {\n        const requests = accessRequestModel.getRequestsByItem('workflow', item.id, 'pending');\n        pendingCount += requests.length;\n      }\n    }\n\n    for (const item of myTrainingPresets) {\n      if (item.access_type === 'request') {\n        const requests = accessRequestModel.getRequestsByItem('training_preset', item.id, 'pending');\n        pendingCount += requests.length;\n      }\n    }\n\n    res.json({ count: pendingCount });\n",
      'pending count training preset loop',
    );
  }

  write(filePath, content);
}

function main() {
  const copyList = [
    'frontend/src/pages/TrainingPresetForm.tsx',
    'frontend/src/pages/TrainingPresetView.tsx',
    'frontend/src/pages/ExternalKeys.tsx',
    'frontend/src/pages/Home.tsx',
    'frontend/src/pages/Profile.tsx',
    'frontend/src/components/TrainingPresetCard.tsx',
    'backend/src/routes/trainingPresets.js',
    'backend/src/models/apiKey.js',
    'backend/src/models/trainingPreset.js',
  ];

  for (const relativePath of copyList) {
    copy(relativePath);
  }

  updateFrontendApp();
  updateFrontendTypes();
  updateFrontendApi();
  updateBackendUserModel();
  updateBackendMiddleware();
  updateBackendAuthRoute();
  updateBackendServer();
  updateBackendPackageJson();
  updateAccessRequestModel();
  updateAccessRequestRoute();

  console.log('sync complete');
}

main();
