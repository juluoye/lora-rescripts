import React, { useMemo, useState } from 'react';
import { Download, Trash2, HardDrive, Gauge, AlertCircle, FolderOpen, CheckSquare, Square } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { useTranslation } from '../hooks/useTranslation';
import { CATEGORY_ORDER } from '../api/types';
import { api } from '../api/bridge';

function formatBytes(bytes: number, language: string) {
  if (!bytes || bytes <= 0) {
    return language === 'zh' ? '0 B' : '0 B';
  }
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 10 || unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

function formatSpeed(bytesPerSecond: number | null | undefined, language: string) {
  if (!bytesPerSecond || bytesPerSecond <= 0) {
    return language === 'zh' ? '等待中' : 'Idle';
  }
  return `${formatBytes(bytesPerSecond, language)}/s`;
}

export function DependenciesPage() {
  const {
    runtimeDefs,
    runtimes,
    dependencyCacheQueue,
    dependencyCacheStates,
    prefetchRuntimeDependencies,
    prefetchRuntimeDependenciesBatch,
    clearRuntimeDependencyCache,
    refreshDependencyCacheStates,
    isInstalling,
    currentTaskState,
    language,
    translations,
  } = useApp();
  const { t } = useTranslation(translations, language);
  const [actionError, setActionError] = useState<string | null>(null);
  const [selectedRuntimeIds, setSelectedRuntimeIds] = useState<string[]>([]);

  const grouped = CATEGORY_ORDER.map((category) => ({
    category,
    label: t(`category_${category}`),
    defs: runtimeDefs.filter((def) => !!dependencyCacheStates[def.id] && def.category === category),
  })).filter((group) => group.defs.length > 0);

  const cacheProgress = useMemo(() => {
    if (currentTaskState.task_type !== 'dependency_cache') {
      return null;
    }
    const details = currentTaskState.details || {};
    return {
      runtimeId: currentTaskState.runtime_id,
      completedItems: Number(details.completed_items || 0),
      totalItems: Number(details.total_items || 0),
      itemLabel: language === 'zh' ? String(details.item_label_zh || '') : String(details.item_label_en || ''),
      speedBytesPerSec: details.item_speed_bytes_per_sec ? Number(details.item_speed_bytes_per_sec) : 0,
      downloadedBytes: details.item_downloaded_bytes ? Number(details.item_downloaded_bytes) : 0,
      totalBytes: details.item_total_bytes == null ? null : Number(details.item_total_bytes),
      state: String(details.state || currentTaskState.state || ''),
    };
  }, [currentTaskState, language]);
  const progressPercent = useMemo(() => {
    if (!cacheProgress || !cacheProgress.totalItems) {
      return 0;
    }
    const done = Math.max(0, cacheProgress.completedItems);
    const currentWeight = cacheProgress.totalBytes && cacheProgress.downloadedBytes
      ? Math.min(1, cacheProgress.downloadedBytes / Math.max(cacheProgress.totalBytes, 1))
      : 0;
    return Math.min(100, ((done + currentWeight) / cacheProgress.totalItems) * 100);
  }, [cacheProgress]);
  const etaSeconds = useMemo(() => {
    if (!cacheProgress || !cacheProgress.totalBytes || !cacheProgress.speedBytesPerSec || cacheProgress.speedBytesPerSec <= 0) {
      return null;
    }
    return Math.max(0, Math.round((cacheProgress.totalBytes - cacheProgress.downloadedBytes) / cacheProgress.speedBytesPerSec));
  }, [cacheProgress]);
  const selectableRuntimeIds = runtimeDefs
    .filter((def) => {
      const runtimeStatus = runtimes[def.id];
      return !!dependencyCacheStates[def.id] && !!runtimeStatus?.python_exists && !!runtimeStatus?.integrity_ok && !!runtimeStatus?.bootstrap_ready;
    })
    .map((def) => def.id);
  const selectedSet = new Set(selectedRuntimeIds);
  const allSelectableChecked = selectableRuntimeIds.length > 0 && selectableRuntimeIds.every((id) => selectedSet.has(id));

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="space-y-2">
        <h2 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>{t('dependencies_page_title')}</h2>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{t('dependencies_page_desc')}</p>
      </div>

      <div className="rounded-2xl p-4 flex flex-wrap items-center justify-between gap-4" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
        <div className="space-y-1">
          <div className="text-sm font-semibold flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
            <Gauge size={16} />
            {t('dependencies_progress_title')}
          </div>
          <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {cacheProgress
              ? t('dependencies_progress_desc', {
                  runtime: cacheProgress.runtimeId || '...',
                  current: String(Math.min(cacheProgress.completedItems + (cacheProgress.state === 'succeeded' ? 0 : 1), Math.max(cacheProgress.totalItems, 1))),
                  total: String(cacheProgress.totalItems || 0),
                  item: cacheProgress.itemLabel || '-',
                })
              : t('dependencies_progress_idle')}
          </div>
          <div className="w-72 max-w-full h-2 rounded-full overflow-hidden mt-3" style={{ backgroundColor: 'var(--bg-input)' }}>
            <div
              className="h-full transition-all duration-300"
              style={{ width: `${progressPercent}%`, backgroundColor: 'var(--accent)' }}
            />
          </div>
        </div>
        <div className="flex flex-wrap gap-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
          <span>{t('dependencies_progress_percent', { percent: String(Math.round(progressPercent)) })}</span>
          <span>{t('dependencies_speed_label')}: {formatSpeed(cacheProgress?.speedBytesPerSec, language)}</span>
          <span>{t('dependencies_downloaded_label')}: {formatBytes(cacheProgress?.downloadedBytes || 0, language)}</span>
          <span>{t('dependencies_eta_label')}: {etaSeconds == null ? t('dependencies_eta_unknown') : t('dependencies_eta_value', { seconds: String(etaSeconds) })}</span>
          <button
            onClick={() => { void refreshDependencyCacheStates(); }}
            className="btn-interactive px-3 py-2 rounded-lg"
            style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)', color: 'var(--text-secondary)' }}
          >
            {t('btn_refresh')}
          </button>
        </div>
      </div>

      {selectableRuntimeIds.length > 0 && (
        <div className="rounded-2xl p-4 space-y-3" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{t('dependencies_batch_title')}</div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {dependencyCacheQueue.active
                  ? t('dependencies_batch_progress', {
                      current: dependencyCacheQueue.current_runtime_id || '...',
                      remaining: String(dependencyCacheQueue.pending_runtime_ids.length),
                    })
                  : t('dependencies_batch_desc')}
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setSelectedRuntimeIds(allSelectableChecked ? [] : selectableRuntimeIds)}
                disabled={isInstalling || dependencyCacheQueue.active}
                className="btn-interactive px-3 py-2 rounded-lg text-xs flex items-center gap-2 disabled:opacity-40"
                style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)', color: 'var(--text-secondary)' }}
              >
                {allSelectableChecked ? <CheckSquare size={14} /> : <Square size={14} />}
                {allSelectableChecked ? t('dependencies_batch_clear') : t('dependencies_batch_select_all')}
              </button>
              <button
                onClick={async () => {
                  setActionError(null);
                  const result = await prefetchRuntimeDependenciesBatch(selectedRuntimeIds);
                  if (result.error) {
                    setActionError(result.error);
                  }
                }}
                disabled={isInstalling || dependencyCacheQueue.active || selectedRuntimeIds.length === 0}
                className="btn-interactive btn-accent-glow px-3 py-2 rounded-lg text-xs flex items-center gap-2 disabled:opacity-40"
                style={{ backgroundColor: 'var(--accent)', color: '#ffffff' }}
              >
                <Download size={14} />
                {t('dependencies_prefetch_selected')}
              </button>
            </div>
          </div>
        </div>
      )}

      {actionError && (
        <div className="rounded-xl p-3 text-sm flex items-start gap-2" style={{ backgroundColor: 'var(--danger-subtle)', border: '1px solid var(--danger-border)', color: 'var(--danger-text)' }}>
          <AlertCircle size={16} />
          <span>{actionError}</span>
        </div>
      )}

      {grouped.map((group) => (
        <div key={group.category} className="space-y-3">
          <h3 className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>{group.label}</h3>
          <div className="space-y-3">
            {group.defs.map((def) => {
              const runtimeStatus = runtimes[def.id];
              const cacheState = dependencyCacheStates[def.id];
              const name = language === 'zh' ? def.name_zh : def.name_en;
              const desc = language === 'zh' ? def.desc_zh : def.desc_en;
              const isActiveCacheTask = currentTaskState.task_type === 'dependency_cache' && currentTaskState.runtime_id === def.id && currentTaskState.state !== 'failed' && currentTaskState.state !== 'succeeded';
              const canCache = !!runtimeStatus?.python_exists && !!runtimeStatus?.integrity_ok && !!runtimeStatus?.bootstrap_ready;
              const isChecked = selectedSet.has(def.id);

              return (
                <div
                  key={def.id}
                  className="rounded-2xl p-4 space-y-4"
                  style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 flex-wrap">
                        {canCache && (
                          <button
                            onClick={() => {
                              setSelectedRuntimeIds((prev) => (
                                prev.includes(def.id)
                                  ? prev.filter((item) => item !== def.id)
                                  : [...prev, def.id]
                              ));
                            }}
                            disabled={isInstalling || dependencyCacheQueue.active}
                            className="btn-interactive w-7 h-7 rounded-lg flex items-center justify-center disabled:opacity-40"
                            style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)', color: 'var(--text-secondary)' }}
                          >
                            {isChecked ? <CheckSquare size={14} /> : <Square size={14} />}
                          </button>
                        )}
                        <h4 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{name}</h4>
                        <span className="text-[10px] px-1.5 py-0.5 rounded" style={{
                          backgroundColor: cacheState?.ready ? 'var(--success-subtle)' : 'var(--warning-subtle)',
                          color: cacheState?.ready ? 'var(--success-text)' : 'var(--warning-text)',
                          border: cacheState?.ready ? '1px solid var(--success-border)' : '1px solid var(--warning-border)',
                        }}>
                          {cacheState?.ready ? t('dependencies_cache_ready') : t('dependencies_cache_partial')}
                        </span>
                        {runtimeStatus?.status_text === 'broken' && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ backgroundColor: 'var(--danger-subtle)', color: 'var(--danger-text)', border: '1px solid var(--danger-border)' }}>
                            {t('status_broken')}
                          </span>
                        )}
                      </div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{desc}</p>
                      <div className="flex flex-wrap gap-4 text-xs" style={{ color: 'var(--text-secondary)' }}>
                        <span className="flex items-center gap-1"><HardDrive size={12} />{t('dependencies_cached_items', { cached: String(cacheState?.cached_items || 0), total: String(cacheState?.total_items || 0) })}</span>
                        <span>{t('dependencies_cached_size', { size: formatBytes(cacheState?.total_bytes || 0, language) })}</span>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={async () => {
                          setActionError(null);
                          const result = await prefetchRuntimeDependencies(def.id);
                          if (result.error) {
                            setActionError(result.error);
                          }
                        }}
                        disabled={isInstalling || !canCache}
                        className="btn-interactive btn-accent-glow px-4 py-2 rounded-lg text-sm text-white disabled:opacity-40"
                        style={{ backgroundColor: 'var(--accent)' }}
                      >
                        <span className="inline-flex items-center gap-2"><Download size={14} />{isActiveCacheTask ? t('dependencies_caching') : t('dependencies_prefetch')}</span>
                      </button>
                      <button
                        onClick={async () => {
                          setActionError(null);
                          const result = await clearRuntimeDependencyCache(def.id);
                          if (result.error) {
                            setActionError(result.error);
                          }
                        }}
                        disabled={isInstalling || !(cacheState?.cache_exists)}
                        className="btn-interactive px-4 py-2 rounded-lg text-sm disabled:opacity-40"
                        style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)', color: 'var(--text-secondary)' }}
                      >
                        <span className="inline-flex items-center gap-2"><Trash2 size={14} />{t('btn_clear_cache')}</span>
                      </button>
                      <button
                        onClick={() => { void api.openPath(cacheState?.cache_dir || ''); }}
                        disabled={!(cacheState?.cache_dir)}
                        className="btn-interactive px-4 py-2 rounded-lg text-sm disabled:opacity-40"
                        style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)', color: 'var(--text-secondary)' }}
                      >
                        <span className="inline-flex items-center gap-2"><FolderOpen size={14} />{t('btn_open_dir')}</span>
                      </button>
                    </div>
                  </div>

                  {!canCache && (
                    <div className="rounded-xl p-3 text-xs" style={{ backgroundColor: 'var(--warning-subtle)', border: '1px solid var(--warning-border)', color: 'var(--warning-text)' }}>
                      {t('dependencies_runtime_not_ready')}
                    </div>
                  )}

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {cacheState?.items.map((item) => (
                      <div key={item.item_id} className="rounded-xl p-3 space-y-2" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
                            {language === 'zh' ? item.label_zh : item.label_en}
                          </div>
                          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{
                            backgroundColor: item.cached ? 'var(--success-subtle)' : 'var(--bg-card)',
                            color: item.cached ? 'var(--success-text)' : 'var(--text-muted)',
                            border: item.cached ? '1px solid var(--success-border)' : '1px solid var(--border-card)',
                          }}>
                            {item.cached ? t('dependencies_cached') : t('dependencies_not_cached')}
                          </span>
                        </div>
                        <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                          {(language === 'zh' ? item.note_zh : item.note_en) || item.kind}
                        </div>
                        <div className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                          {t('dependencies_item_meta', {
                            files: String(item.file_count),
                            size: formatBytes(item.bytes, language),
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
