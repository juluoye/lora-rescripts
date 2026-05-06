import React, { useEffect, useState } from 'react';
import { Download, CheckCircle2, AlertCircle, XCircle, Sparkles, Trash2, ShieldAlert, X } from 'lucide-react';
import { createPortal } from 'react-dom';
import { CompatibilitySummary } from '../components/CompatibilitySummary';
import { useApp } from '../context/AppContext';
import { useTranslation } from '../hooks/useTranslation';
import { CATEGORY_ORDER } from '../api/types';
import type { RuntimeCapabilityTag, RuntimeStatus } from '../api/types';

function StatusBadge({ status }: { status: RuntimeStatus }) {
  const { language, translations } = useApp();
  const { t } = useTranslation(translations, language);

  const statusConfig = {
    installed: { label: t('status_installed'), Icon: CheckCircle2 as React.FC<{ size?: number }> },
    initialized: { label: t('status_initialized'), Icon: AlertCircle as React.FC<{ size?: number }> },
    broken: { label: t('status_broken'), Icon: XCircle as React.FC<{ size?: number }> },
    partial: { label: t('status_partial'), Icon: AlertCircle as React.FC<{ size?: number }> },
    missing: { label: t('status_missing'), Icon: XCircle as React.FC<{ size?: number }> },
  }[status.status_text] ?? { label: status.status_text, Icon: XCircle as React.FC<{ size?: number }> };

  const colorMap: Record<string, { bg: string; text: string }> = {
    installed: { bg: 'var(--success-subtle)', text: 'var(--success-text)' },
    initialized: { bg: 'var(--warning-subtle)', text: 'var(--warning-text)' },
    broken: { bg: 'var(--danger-subtle)', text: 'var(--danger-text)' },
    partial: { bg: 'var(--warning-subtle)', text: 'var(--warning-text)' },
    missing: { bg: 'var(--bg-card)', text: 'var(--text-muted)' },
  };
  const colors = colorMap[status.status_text] || colorMap.missing;

  const StatusIcon = statusConfig.Icon;

  return (
    <span className="text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1" style={{ backgroundColor: colors.bg, color: colors.text }}>
      <StatusIcon size={10} />
      {statusConfig.label}
    </span>
  );
}

function CapabilityTagList({
  tags,
  language,
}: {
  tags: RuntimeCapabilityTag[];
  language: string;
}) {
  const toneMap: Record<RuntimeCapabilityTag['tone'], { bg: string; text: string; border: string }> = {
    success: { bg: 'var(--success-subtle)', text: 'var(--success-text)', border: 'var(--success-border)' },
    accent: { bg: 'var(--accent-subtle)', text: 'var(--accent-text)', border: 'var(--accent-border)' },
    warning: { bg: 'var(--warning-subtle)', text: 'var(--warning-text)', border: 'var(--warning-border)' },
  };

  if (tags.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-wrap gap-2">
      {tags.map((tag) => {
        const colors = toneMap[tag.tone] || toneMap.accent;
        return (
          <span
            key={tag.id}
            className="text-[10px] px-2 py-1 rounded-full"
            style={{ backgroundColor: colors.bg, color: colors.text, border: `1px solid ${colors.border}` }}
          >
            {language === 'zh' ? tag.label_zh : tag.label_en}
          </span>
        );
      })}
    </div>
  );
}

function ConfirmDialog({
  open,
  title,
  message,
  runtimeName,
  language,
  confirmLabel,
  cancelLabel,
  busy,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  message: string;
  runtimeName: string;
  language: string;
  confirmLabel: string;
  cancelLabel: string;
  busy: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  useEffect(() => {
    if (!open) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !busy) {
        event.preventDefault();
        onCancel();
      }
      if (event.key === 'Enter' && !busy) {
        event.preventDefault();
        onConfirm();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [busy, onCancel, onConfirm, open]);

  if (!open) {
    return null;
  }

  const helperBadge = language === 'zh' ? '依赖级卸载' : 'Dependency-only';
  const helperTitle = language === 'zh' ? '这次不会删除整个运行时目录' : 'This will not remove the whole runtime directory';
  const helperDesc = language === 'zh'
    ? '会保留本地 Python、pip bootstrap 和初始化骨架，后续可以直接重新安装依赖。'
    : 'The local Python, pip bootstrap, and initialized runtime skeleton will stay intact so you can reinstall dependencies directly later.';

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4"
      style={{ backdropFilter: 'blur(8px)' }}
      onClick={() => {
        if (!busy) {
          onCancel();
        }
      }}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="w-full max-w-xl rounded-2xl shadow-2xl"
        style={{
          backgroundColor: 'var(--bg-base)',
          border: '1px solid var(--border-card)',
          boxShadow: '0 24px 80px rgba(0, 0, 0, 0.28)',
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="px-6 pt-6">
          <div className="flex items-start justify-between gap-4">
            <div className="flex min-w-0 items-start gap-4">
              <div
                className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl"
                style={{
                  backgroundColor: 'var(--danger-subtle)',
                  color: 'var(--danger-text)',
                  border: '1px solid var(--danger-border)',
                }}
              >
                <ShieldAlert size={20} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {title}
                </div>
                <p className="mt-2 text-sm leading-6" style={{ color: 'var(--text-secondary)' }}>
                  {message}
                </p>
              </div>
            </div>
            <button
              onClick={onCancel}
              disabled={busy}
              className="btn-interactive rounded-xl p-2 disabled:opacity-40"
              style={{ backgroundColor: 'var(--bg-input)', color: 'var(--text-muted)' }}
              aria-label={cancelLabel}
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="px-6 pt-5">
          <div
            className="rounded-2xl p-4"
            style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span
                className="text-[10px] px-2 py-1 rounded-full"
                style={{
                  backgroundColor: 'var(--accent-subtle)',
                  color: 'var(--accent-text)',
                  border: '1px solid var(--accent-border)',
                }}
              >
                {runtimeName}
              </span>
              <span
                className="text-[10px] px-2 py-1 rounded-full"
                style={{
                  backgroundColor: 'var(--danger-subtle)',
                  color: 'var(--danger-text)',
                  border: '1px solid var(--danger-border)',
                }}
              >
                {helperBadge}
              </span>
            </div>
            <div className="mt-3 text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
              {helperTitle}
            </div>
            <div className="mt-2 text-xs leading-6" style={{ color: 'var(--text-secondary)' }}>
              {helperDesc}
            </div>
          </div>
        </div>

        <div
          className="mt-6 flex justify-end gap-3 border-t px-6 py-4"
          style={{ borderColor: 'var(--border-card)' }}
        >
          <button
            onClick={onCancel}
            disabled={busy}
            className="btn-interactive px-4 py-2 rounded-xl text-sm font-medium disabled:opacity-40"
            style={{
              backgroundColor: 'var(--bg-input)',
              color: 'var(--text-secondary)',
              border: '1px solid var(--border-card)',
            }}
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            disabled={busy}
            className="btn-interactive px-4 py-2 rounded-xl text-sm font-medium text-white disabled:opacity-40"
            style={{
              backgroundColor: 'var(--danger)',
              boxShadow: '0 10px 24px color-mix(in srgb, var(--danger) 35%, transparent)',
            }}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

export function RuntimePage() {
  const {
    runtimes,
    runtimeDefs,
    selectedRuntime,
    selectRuntime,
    initializeRuntime,
    installRuntime,
    uninstallRuntime,
    isInstalling,
    currentTaskState,
    runtimeRecommendation,
    runtimeCompatibility,
    language,
    translations,
  } = useApp();
  const { t } = useTranslation(translations, language);
  const [actionError, setActionError] = useState<string | null>(null);
  const [confirmingRuntime, setConfirmingRuntime] = useState<{ id: string; name: string } | null>(null);
  const isInitializingTask = currentTaskState.task_type === 'initialize';

  const grouped = CATEGORY_ORDER.map((cat) => ({
    category: cat,
    label: t(`category_${cat}`),
    defs: runtimeDefs.filter((d) => d.category === cat),
  })).filter((g) => g.defs.length > 0);
  const selectedDef = selectedRuntime ? runtimeDefs.find((item) => item.id === selectedRuntime) || null : null;
  const selectedStatus = selectedRuntime ? runtimes[selectedRuntime] || null : null;
  const selectedRuntimeName = selectedDef
    ? (language === 'zh' ? selectedDef.name_zh : selectedDef.name_en)
    : null;
  const selectedIntegrityMessage = selectedStatus
    ? (language === 'zh' ? selectedStatus.integrity_message_zh : selectedStatus.integrity_message_en)
    : null;

  const handleConfirmUninstall = async () => {
    if (!confirmingRuntime) {
      return;
    }

    const target = confirmingRuntime;
    setConfirmingRuntime(null);
    const result = await uninstallRuntime(target.id);
    if (result.error) {
      setActionError(result.error);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in animate-slide-in-right">
      <ConfirmDialog
        open={!!confirmingRuntime}
        title={t('runtime_uninstall_dialog_title')}
        message={confirmingRuntime ? t('runtime_uninstall_confirm', { runtime: confirmingRuntime.name }) : ''}
        runtimeName={confirmingRuntime?.name || ''}
        language={language}
        confirmLabel={t('runtime_uninstall_confirm_action')}
        cancelLabel={t('runtime_uninstall_cancel')}
        busy={isInstalling}
        onConfirm={handleConfirmUninstall}
        onCancel={() => setConfirmingRuntime(null)}
      />
      {actionError && (
        <div className="rounded-xl p-3 text-sm flex items-start gap-2" style={{ backgroundColor: 'var(--danger-subtle)', border: '1px solid var(--danger-border)', color: 'var(--danger-text)' }}>
          <AlertCircle size={16} />
          <span>{actionError}</span>
        </div>
      )}

      {runtimeRecommendation && (
        <div className="rounded-2xl p-5" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
          <div className="flex items-start gap-3">
            <Sparkles size={18} style={{ color: 'var(--accent-text)' }} />
            <div>
              <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                {t('runtime_recommendation_title')}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {language === 'zh' ? runtimeRecommendation.reason_zh : runtimeRecommendation.reason_en}
              </div>
              {runtimeRecommendation.gpu_name && (
                <div className="text-xs mt-2" style={{ color: 'var(--text-secondary)' }}>
                  {t('runtime_detected_gpu', { gpu: runtimeRecommendation.gpu_name })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {selectedDef && selectedStatus && (
        <div className="rounded-2xl p-5" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
          <div className="flex items-start justify-between gap-4 mb-4">
            <div>
              <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                {t('runtime_details_title')}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {selectedRuntimeName || selectedRuntime}
              </div>
            </div>
            <StatusBadge status={selectedStatus} />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
              <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{t('runtime_status_label')}</div>
              <div className="text-sm font-medium mt-1" style={{ color: 'var(--text-primary)' }}>{t(`status_${selectedStatus.status_text}`)}</div>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
              <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{t('runtime_integrity_label')}</div>
              <div className="text-sm font-medium mt-1" style={{ color: selectedStatus.integrity_ok ? 'var(--success-text)' : 'var(--danger-text)' }}>
                {selectedStatus.integrity_ok ? t('runtime_integrity_ok') : t('runtime_integrity_problem')}
              </div>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
              <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{t('runtime_expected_dirs')}</div>
              <div className="text-xs font-mono mt-1 break-all" style={{ color: 'var(--text-primary)' }}>
                {selectedDef.env_dir_names.map((name) => `.\\env\\${name}`).join(', ')}
              </div>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
              <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{t('runtime_bootstrap_label')}</div>
              <div className="text-sm font-medium mt-1" style={{ color: selectedStatus.bootstrap_ready ? 'var(--success-text)' : 'var(--warning-text)' }}>
                {selectedStatus.bootstrap_ready ? t('runtime_bootstrap_ready') : t('runtime_bootstrap_missing')}
              </div>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
              <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{t('runtime_env_dir')}</div>
              <div className="text-xs font-mono mt-1 break-all" style={{ color: 'var(--text-primary)' }}>
                {selectedStatus.env_dir || t('runtime_details_missing')}
              </div>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
              <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{t('runtime_python_path')}</div>
              <div className="text-xs font-mono mt-1 break-all" style={{ color: 'var(--text-primary)' }}>
                {selectedStatus.python_path || t('runtime_details_missing')}
              </div>
            </div>
            <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
              <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                {language === 'zh' ? '启动入口' : 'Launch entry'}
              </div>
              <div className="text-xs font-mono mt-1 break-all" style={{ color: 'var(--text-primary)' }}>
                {`${selectedDef.launch_entry.script} (${selectedDef.launch_entry.mode})`}
              </div>
            </div>
            <div className="rounded-xl p-3 col-span-2" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
              <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{t('runtime_install_scripts')}</div>
              <div className="text-xs font-mono mt-1 break-all" style={{ color: 'var(--text-primary)' }}>
                {selectedDef.install_scripts.length > 0 ? selectedDef.install_scripts.join(', ') : t('runtime_details_missing')}
              </div>
            </div>
            {selectedDef.runtime_env_vars.length > 0 && (
              <div className="rounded-xl p-3 col-span-2" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
                <div className="text-[11px] mb-2" style={{ color: 'var(--text-muted)' }}>
                  {language === 'zh' ? '运行时默认环境变量' : 'Runtime default env vars'}
                </div>
                <div className="flex flex-wrap gap-2">
                  {selectedDef.runtime_env_vars.map((entry) => (
                    <span
                      key={entry.key}
                      className="text-[10px] px-2 py-1 rounded"
                      style={{ backgroundColor: 'var(--accent-subtle)', color: 'var(--accent-text)', border: '1px solid var(--accent-border)' }}
                    >
                      {`${entry.key}=${entry.value}`}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {(selectedDef.capability_tags || []).length > 0 && (
              <div className="rounded-xl p-3 col-span-2" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
                <div className="text-[11px] mb-2" style={{ color: 'var(--text-muted)' }}>{t('runtime_capability_tags')}</div>
                <CapabilityTagList tags={selectedDef.capability_tags || []} language={language} />
              </div>
            )}
            {selectedIntegrityMessage && (
              <div className="rounded-xl p-3 col-span-2" style={{ backgroundColor: selectedStatus.integrity_ok ? 'var(--bg-input)' : 'var(--danger-subtle)', border: `1px solid ${selectedStatus.integrity_ok ? 'var(--border-card)' : 'var(--danger-border)'}` }}>
                <div className="text-[11px] mb-2" style={{ color: 'var(--text-muted)' }}>
                  {t('runtime_integrity_message')}
                </div>
                <div className="text-xs leading-relaxed" style={{ color: selectedStatus.integrity_ok ? 'var(--text-secondary)' : 'var(--danger-text)' }}>
                  {selectedIntegrityMessage}
                </div>
              </div>
            )}
            {(language === 'zh' ? selectedDef.notes_zh : selectedDef.notes_en) && (
              <div className="rounded-xl p-3 col-span-2" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
                <div className="text-[11px] mb-2" style={{ color: 'var(--text-muted)' }}>
                  {language === 'zh' ? '运行时备注' : 'Runtime note'}
                </div>
                <div className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                  {language === 'zh' ? selectedDef.notes_zh : selectedDef.notes_en}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {grouped.map((group) => (
        <div key={group.category}>
          <h3 className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>
            {group.label}
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {group.defs.map((def) => {
              const status = runtimes[def.id];
              const isSelected = selectedRuntime === def.id;
              const name = language === 'zh' ? def.name_zh : def.name_en;
              const desc = language === 'zh' ? def.desc_zh : def.desc_en;
              const compatibilityEntries = runtimeCompatibility[def.id] || [];
              const capabilityTags = def.capability_tags || [];
              const runtimePathHint = def.env_dir_names.length > 0 ? `.\\env\\${def.env_dir_names[0]}` : '.\\env';
              const isInitialized = status?.status_text === 'initialized';
              const needsInitialize = !status?.python_exists || !status?.integrity_ok || !status?.bootstrap_ready;
              const isInstalled = Boolean(status?.installed);
              const isBusy = isInstalling;
              const installHint = status?.installed
                ? null
                : status?.status_text === 'broken'
                  ? t('install_broken_runtime_hint', { dir: runtimePathHint })
                : isInitialized
                  ? t('install_ready_runtime_hint')
                  : status?.status_text === 'partial'
                    ? t('install_incomplete_runtime_hint', { dir: runtimePathHint })
                    : t('install_prepare_runtime_hint', { dir: runtimePathHint });
              const actionLabel = isInstalling
                ? isInitializingTask
                  ? t('btn_initializing')
                  : t('btn_installing')
                : needsInitialize
                  ? t('btn_initialize')
                  : t('btn_install');

              return (
                <div
                  key={def.id}
                  className={`card-interactive p-5 rounded-2xl`}
                  style={{
                    backgroundColor: isSelected ? 'var(--accent-subtle)' : 'var(--bg-card)',
                    border: isSelected ? '1px solid var(--accent-border)' : '1px solid var(--border-card)',
                    boxShadow: isSelected ? '0 0 0 1px var(--accent-ring)' : 'none',
                  }}
                >
                  <div className="flex justify-between items-start mb-3">
                    <div className="w-10 h-10 rounded-lg flex items-center justify-center font-bold border" style={{ backgroundColor: 'var(--bg-input)', color: 'var(--accent-text)', borderColor: 'var(--border-card)' }}>
                      {name.charAt(0)}
                    </div>
                    <div className="flex items-center gap-2">
                      {status && <StatusBadge status={status} />}
                    </div>
                  </div>
                  <h3 className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
                    {name}
                    {def.experimental && (
                      <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded" style={{ backgroundColor: 'var(--warning-subtle)', color: 'var(--warning-text)', border: '1px solid var(--warning-border)' }}>
                        {t('experimental_badge')}
                      </span>
                    )}
                    {runtimeRecommendation?.selected_runtime_id === def.id && (
                      <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded" style={{ backgroundColor: 'var(--accent-subtle)', color: 'var(--accent-text)', border: '1px solid var(--accent-border)' }}>
                        {t('runtime_recommended_badge')}
                      </span>
                    )}
                    {runtimeRecommendation?.preferred_runtime_id === def.id && runtimeRecommendation.selected_runtime_id !== def.id && (
                      <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded" style={{ backgroundColor: 'var(--success-subtle)', color: 'var(--success-text)', border: '1px solid var(--success-border)' }}>
                        {t('runtime_preferred_badge')}
                      </span>
                    )}
                  </h3>
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>{desc}</p>
                  {capabilityTags.length > 0 && (
                    <div className="mt-3">
                      <CapabilityTagList tags={capabilityTags} language={language} />
                    </div>
                  )}
                  {installHint && (
                    <p className="text-xs mt-2" style={{ color: 'var(--warning-text)' }}>{installHint}</p>
                  )}
                  <div className="mt-4 flex flex-wrap gap-2">
                    {isInstalled ? (
                      <>
                        <button
                          onClick={async () => {
                            setActionError(null);
                            await selectRuntime(def.id);
                          }}
                          disabled={isBusy || isSelected}
                          className={`btn-interactive flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium disabled:opacity-40`}
                          style={{
                            backgroundColor: isSelected ? 'var(--success)' : 'var(--accent)',
                            color: '#ffffff',
                          }}
                        >
                          <CheckCircle2 size={14} />
                          {isSelected ? t('btn_runtime_enabled') : t('btn_runtime_enable')}
                        </button>
                        <button
                          onClick={async () => {
                            setActionError(null);
                            setConfirmingRuntime({ id: def.id, name });
                          }}
                          disabled={isBusy}
                          className="btn-interactive flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium disabled:opacity-40"
                          style={{
                            backgroundColor: 'var(--danger-subtle)',
                            color: 'var(--danger-text)',
                            border: '1px solid var(--danger-border)',
                          }}
                        >
                          <Trash2 size={14} />
                          {t('btn_runtime_uninstall')}
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={async () => {
                          setActionError(null);
                          const result = needsInitialize
                            ? await initializeRuntime(def.id)
                            : await installRuntime(def.id);
                          if (result.error) {
                            setActionError(result.error);
                          }
                        }}
                        disabled={isBusy}
                        className={`btn-interactive ${isBusy ? 'btn-shimmer' : 'btn-accent-glow'} flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium disabled:opacity-40`}
                        style={{ backgroundColor: 'var(--accent)', color: '#ffffff' }}
                      >
                        <Download size={14} />
                        {actionLabel}
                      </button>
                    )}
                  </div>
                  {compatibilityEntries.length > 0 && (
                    <div className="mt-4 pt-4" style={{ borderTop: '1px solid var(--border-card)' }}>
                      <CompatibilitySummary
                        entries={compatibilityEntries}
                        language={language}
                        translations={translations}
                        compact
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
