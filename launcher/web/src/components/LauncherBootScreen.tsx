import React, { useMemo } from 'react';
import launcherLogo from '../../../assets/favicon-launcher.png';
import type { BootStage, Theme } from '../context/AppContext';

type Language = 'zh' | 'en';
type GpuVendor = 'nvidia' | 'amd' | 'intel' | 'unknown';
type BootMessageKey =
  | 'fox_den'
  | 'workflow_magic'
  | 'training_engine'
  | 'gpu_warmup'
  | 'gpu_calm'
  | 'vram_medicine'
  | 'fan_takeoff'
  | 'vram_negotiation'
  | 'gpu_honor'
  | 'electron_fragrance'
  | 'ceo_prayer'
  | 'tail_fluffiness'
  | 'data_shredder'
  | 'tag_discipline'
  | 'overfit_edge'
  | 'pixel_soul'
  | 'regularization_powder'
  | 'why_images_like_this'
  | 'gradient_cooling'
  | 'alchemy_scrolls'
  | 'dont_blow_up'
  | 'perceptron_rebellion'
  | 'fox_grooming'
  | 'loss_curve'
  | 'failure_excuses'
  | 'poetry_in_error'
  | 'pretend_understand'
  | 'power_outage'
  | 'bug_hidden_feature'
  | 'sleeping_networks';

interface LauncherBootScreenProps {
  visible: boolean;
  stage: BootStage;
  language: string;
  version: string;
  theme: Theme;
  error?: string | null;
  gpuVendor?: string | null;
}

function getText(language: string, mapping: { zh: string; en: string }) {
  return language === 'zh' ? mapping.zh : mapping.en;
}

function normalizeGpuVendor(value?: string | null): GpuVendor {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized.includes('nvidia')) return 'nvidia';
  if (normalized.includes('amd')) return 'amd';
  if (normalized.includes('intel')) return 'intel';
  return 'unknown';
}

function shuffleKeys<T>(items: T[]): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

const ZH_BOOT_POOL: BootMessageKey[] = [
  'fox_den',
  'workflow_magic',
  'training_engine',
  'gpu_warmup',
  'gpu_calm',
  'vram_medicine',
  'fan_takeoff',
  'vram_negotiation',
  'gpu_honor',
  'electron_fragrance',
  'ceo_prayer',
  'tail_fluffiness',
  'data_shredder',
  'tag_discipline',
  'overfit_edge',
  'pixel_soul',
  'regularization_powder',
  'why_images_like_this',
  'gradient_cooling',
  'alchemy_scrolls',
  'dont_blow_up',
  'perceptron_rebellion',
  'fox_grooming',
  'loss_curve',
  'failure_excuses',
  'poetry_in_error',
  'pretend_understand',
  'power_outage',
  'bug_hidden_feature',
  'sleeping_networks',
];

const EN_BOOT_POOL = [
  'Cleaning the fox den... 🦊',
  'Weaving workflow magic circles... ✨',
  'Warming up the training engine... 🚂',
  'Doing warm-up exercises for the GPU... 💪',
  'Persuading the fan not to achieve liftoff...',
  'Negotiating with overflowing VRAM...',
  'Injecting a soul into every pixel...',
  'Cooling gradient descent by physical means...',
  'Pretending we totally understand this code...',
  'Turning bugs into hidden features...',
];

const STAGE_SEQUENCE: BootStage[] = [
  'waiting_api',
  'loading_config',
  'loading_runtime',
  'finalizing',
  'ready',
];

function getPreferredCeoAlias(vendor: GpuVendor): string {
  if (vendor === 'nvidia') return '黄皮衣';
  if (vendor === 'amd') return '苏妈';
  if (vendor === 'intel') return 'Mr.陈';
  return '显卡之神';
}

function getOtherCeoAlias(vendor: GpuVendor): string {
  const aliases = ['黄皮衣', '苏妈', 'Mr.陈'];
  const preferred = getPreferredCeoAlias(vendor);
  const others = aliases.filter((alias) => alias !== preferred);
  return others[Math.floor(Math.random() * others.length)] || preferred;
}

function resolveZhMessage(key: BootMessageKey, vendor: GpuVendor): string {
  switch (key) {
    case 'fox_den':
      return '正在打扫狐狸窝... 🦊';
    case 'workflow_magic':
      return '正在编织工作流魔法阵... ✨';
    case 'training_engine':
      return '正在热身训练引擎... 🚂';
    case 'gpu_warmup':
      return '正在给 GPU 做热身操...';
    case 'gpu_calm':
      return '正在哄骗 GPU 保持冷静...';
    case 'vram_medicine':
      return '正在给显存喂降压药...';
    case 'fan_takeoff':
      return '正在劝说风扇不要起飞...';
    case 'vram_negotiation':
      return '正在和溢出的显存进行艰难谈判...';
    case 'gpu_honor':
      return '正在为显卡的尊严而战...';
    case 'electron_fragrance':
      return '正在计算烧焦空气中的电子芬芳...';
    case 'ceo_prayer': {
      const alias = Math.random() < 0.8 ? getPreferredCeoAlias(vendor) : getOtherCeoAlias(vendor);
      return `正在向${alias}祈祷不爆显存...`;
    }
    case 'tail_fluffiness':
      return '正在校准尾巴的蓬松程度参数...';
    case 'data_shredder':
      return '正在把脏数据丢进碎纸机...';
    case 'tag_discipline':
      return '正在强行纠正标签的散漫作风...';
    case 'overfit_edge':
      return '正在过拟合的边缘反复试探...';
    case 'pixel_soul':
      return '正在给每一个像素点注入灵魂...';
    case 'regularization_powder':
      return '正在往坩埚里添加少许正则化粉末...';
    case 'why_images_like_this':
      return '正在试图理解这些图片为什么长这样...';
    case 'gradient_cooling':
      return '正在对梯度下降进行物理降温...';
    case 'alchemy_scrolls':
      return '正在翻找前人留下的炼丹残卷...';
    case 'dont_blow_up':
      return '正在默念：不炸锅、不崩盘、不发灰...';
    case 'perceptron_rebellion':
      return '正在微调感知机的叛逆心理...';
    case 'fox_grooming':
      return '正在帮小狐狸梳毛...';
    case 'loss_curve':
      return '正在等待 Loss 曲线浪子回头...';
    case 'failure_excuses':
      return '正在编造训练失败后的借口...';
    case 'poetry_in_error':
      return '正在试图从报错信息中读出诗意...';
    case 'pretend_understand':
      return '正在假装自己知道自己在跑什么代码...';
    case 'power_outage':
      return '正在祈祷停电不要发生在这个瞬间...';
    case 'bug_hidden_feature':
      return '正在把 Bug 重新定义为“隐藏特性”...';
    case 'sleeping_networks':
      return '正在唤醒沉睡的神经网络... 🧠';
    default:
      return '正在打扫狐狸窝... 🦊';
  }
}

export function LauncherBootScreen({
  visible,
  stage,
  language,
  version,
  theme,
  error,
  gpuVendor,
}: LauncherBootScreenProps) {
  const normalizedLanguage: Language = language === 'en' ? 'en' : 'zh';
  const accentTone = theme === 'light' ? 'var(--accent)' : 'var(--accent-light)';
  const normalizedVendor = normalizeGpuVendor(gpuVendor);
  const progressWidth = useMemo(() => {
    switch (stage) {
      case 'waiting_api':
        return 12;
      case 'loading_config':
        return 36;
      case 'loading_runtime':
        return 68;
      case 'finalizing':
        return 88;
      case 'ready':
        return 100;
      case 'error':
      default:
        return 100;
    }
  }, [stage]);
  const sampledMessages = useMemo(() => {
    if (normalizedLanguage === 'en') {
      return shuffleKeys(EN_BOOT_POOL).slice(0, STAGE_SEQUENCE.length);
    }
    return shuffleKeys(ZH_BOOT_POOL).slice(0, STAGE_SEQUENCE.length);
  }, [normalizedLanguage]);

  const funMessage = useMemo(() => {
    if (stage === 'error') {
      return normalizedLanguage === 'zh'
        ? '启动流程卡住了，稍后重启启动器再试试。'
        : 'Startup got stuck. Please restart the launcher and try again.';
    }
    const index = STAGE_SEQUENCE.indexOf(stage);
    if (index < 0) {
      return normalizedLanguage === 'zh' ? '正在打扫狐狸窝... 🦊' : 'Cleaning the fox den... 🦊';
    }
    if (normalizedLanguage === 'en') {
      return sampledMessages[index] || EN_BOOT_POOL[0];
    }
    return resolveZhMessage((sampledMessages[index] as BootMessageKey) || 'fox_den', normalizedVendor);
  }, [normalizedLanguage, normalizedVendor, sampledMessages, stage]);

  return (
    <div
      className={`launcher-boot-screen fixed inset-0 z-50 transition-opacity duration-500 ${visible ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}
      style={{
        background:
          theme === 'light'
            ? 'radial-gradient(circle at top right, rgba(217, 119, 6, 0.12), transparent 34%), radial-gradient(circle at bottom left, rgba(139, 92, 246, 0.10), transparent 28%), #f5f5f4'
            : 'radial-gradient(circle at top right, rgba(37, 99, 235, 0.18), transparent 34%), radial-gradient(circle at bottom left, rgba(168, 85, 247, 0.14), transparent 28%), #0a0b0e',
      }}
      aria-hidden={!visible}
    >
      <div className="absolute inset-0 overflow-hidden">
        <div
          className="absolute -top-16 -right-20 h-64 w-64 rounded-full blur-3xl"
          style={{ backgroundColor: theme === 'light' ? 'rgba(217, 119, 6, 0.18)' : 'rgba(37, 99, 235, 0.22)' }}
        />
        <div
          className="absolute -bottom-20 -left-12 h-56 w-56 rounded-full blur-3xl"
          style={{ backgroundColor: theme === 'light' ? 'rgba(139, 92, 246, 0.12)' : 'rgba(168, 85, 247, 0.18)' }}
        />
        <div className="launcher-boot-grid absolute inset-0 opacity-70" />
      </div>

      <div className="relative z-10 flex h-full items-center justify-center px-6">
        <div
          className="w-full max-w-3xl rounded-[28px] border px-8 py-8 shadow-2xl backdrop-blur-xl md:px-10 md:py-10"
          style={{
            backgroundColor: theme === 'light' ? 'rgba(255, 255, 255, 0.72)' : 'rgba(10, 11, 14, 0.72)',
            borderColor: theme === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.08)',
            boxShadow: theme === 'light' ? '0 24px 80px rgba(0, 0, 0, 0.10)' : '0 24px 80px rgba(0, 0, 0, 0.35)',
          }}
        >
          <div className="flex flex-col gap-8 md:flex-row md:items-center md:justify-between">
            <div className="max-w-xl">
              <div className="flex items-center gap-4">
                <div
                  className="relative h-16 w-16 overflow-hidden rounded-2xl border"
                  style={{
                    borderColor: theme === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.10)',
                    backgroundColor: theme === 'light' ? 'rgba(255, 255, 255, 0.86)' : 'rgba(255, 255, 255, 0.06)',
                  }}
                >
                  <img src={launcherLogo} alt="Launcher" className="h-full w-full object-cover" />
                  <div
                    className="absolute inset-0 rounded-2xl"
                    style={{ boxShadow: `inset 0 0 0 1px ${theme === 'light' ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.10)'}` }}
                  />
                </div>
                <div>
                  <div className="text-[11px] uppercase tracking-[0.35em]" style={{ color: 'var(--text-muted)' }}>
                    SD-RESCRIPTS
                  </div>
                  <h1 className="mt-2 text-3xl font-black tracking-tight md:text-4xl" style={{ color: 'var(--text-primary)' }}>
                    {normalizedLanguage === 'zh' ? '启动器准备中' : 'Launcher Booting'}
                  </h1>
                </div>
              </div>

              <div className="mt-6">
                <p className="max-w-xl text-lg leading-7" style={{ color: stage === 'error' ? 'var(--danger-text)' : 'var(--text-secondary)' }}>
                  {funMessage}
                </p>
                {stage === 'error' && error ? (
                  <div
                    className="mt-4 rounded-2xl border px-4 py-3 text-xs leading-5"
                    style={{
                      backgroundColor: 'var(--danger-subtle)',
                      borderColor: 'var(--danger-border)',
                      color: 'var(--danger-text)',
                    }}
                  >
                    {error}
                  </div>
                ) : null}
              </div>
            </div>

            <div className="flex min-w-[220px] flex-col items-start gap-4 md:items-end">
              <div className="launcher-boot-spinner" style={{ ['--boot-accent' as string]: accentTone } as React.CSSProperties} />
              <div
                className="rounded-full border px-3 py-1 text-[11px] font-mono uppercase tracking-[0.22em]"
                style={{
                  color: 'var(--text-muted)',
                  borderColor: theme === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.08)',
                  backgroundColor: theme === 'light' ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.04)',
                }}
              >
                {version || 'v1.6.24'}
              </div>
            </div>
          </div>

          <div className="mt-8">
            <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.26em]" style={{ color: 'var(--text-muted)' }}>
              <span>{normalizedLanguage === 'zh' ? '启动进度' : 'Boot Progress'}</span>
              <span>{stage === 'error' ? 'ERR' : `${progressWidth}%`}</span>
            </div>
            <div
              className="mt-3 h-2.5 overflow-hidden rounded-full"
              style={{ backgroundColor: theme === 'light' ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.07)' }}
            >
              <div
                className={`h-full rounded-full transition-all duration-500 ${stage === 'error' ? '' : 'launcher-boot-progress'}`}
                style={{
                  width: `${progressWidth}%`,
                  background: stage === 'error'
                    ? 'linear-gradient(90deg, rgba(239,68,68,0.95), rgba(248,113,113,0.9))'
                    : `linear-gradient(90deg, ${theme === 'light' ? 'rgba(217,119,6,0.95), rgba(245,158,11,0.92), rgba(168,85,247,0.88)' : 'rgba(37,99,235,0.95), rgba(59,130,246,0.92), rgba(168,85,247,0.9)'})`,
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
