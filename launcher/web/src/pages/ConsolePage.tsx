import React, { useRef, useEffect, useMemo, useState } from 'react';
import { Copy, Trash2, ArrowRight, CheckCircle2, AlertTriangle, Download, Square, ChevronDown, ChevronUp } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { useTranslation } from '../hooks/useTranslation';
import { colorizeLine } from '../utils/consoleColor';
import type { TaskCommandRecord, TaskLogSignal, TaskResultRecord } from '../api/types';

export function ConsolePage() {
  const {
    consoleLines,
    isRunning,
    currentTaskState,
    taskStageEvents,
    taskHistory,
    clearConsole,
    clearTaskHistory,
    stop,
    lastInstallSummary,
    clearInstallSummary,
    runtimeDefs,
    setActivePage,
    language,
    translations,
  } = useApp();
  const { t } = useTranslation(translations, language);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [showCurrentTaskStages, setShowCurrentTaskStages] = useState(false);
  const [showRecentTasks, setShowRecentTasks] = useState(false);
  const [showTaskDetails, setShowTaskDetails] = useState(false);
  const installedRuntimeName = useMemo(() => {
    if (!lastInstallSummary) return null;
    const matched = runtimeDefs.find((item) => item.id === lastInstallSummary.runtimeId);
    if (!matched) return lastInstallSummary.runtimeId;
    return language === 'zh' ? matched.name_zh : matched.name_en;
  }, [lastInstallSummary, runtimeDefs, language]);
  const isStoppingTask = isTaskStopping(currentTaskState.task_type, currentTaskState.state);
  const installProgress = useMemo(
    () => buildInstallProgressState(currentTaskState, runtimeDefs, language, t),
    [currentTaskState, runtimeDefs, language, t],
  );
  const [displayInstallProgressPercent, setDisplayInstallProgressPercent] = useState(0);
  const installProgressTaskRef = useRef<string | null>(null);
  const installProgressSectionRef = useRef<string>('');
  const recentTaskStages = useMemo(() => [...taskStageEvents].slice(-8).reverse(), [taskStageEvents]);
  const liveTaskRecord = useMemo(
    () => buildLiveTaskRecord(currentTaskState, taskStageEvents, consoleLines),
    [currentTaskState, taskStageEvents, consoleLines],
  );
  const recentTaskHistory = useMemo(() => {
    const recentHistory = taskHistory.slice(0, 6);
    if (!liveTaskRecord?.task_id) {
      return recentHistory;
    }
    const next = [liveTaskRecord, ...recentHistory.filter((task) => task.task_id !== liveTaskRecord.task_id)];
    return next.slice(0, 6);
  }, [taskHistory, liveTaskRecord]);
  const selectedTask = useMemo(() => {
    if (recentTaskHistory.length === 0) return null;
    if (selectedTaskId) {
      const matched = recentTaskHistory.find((task) => getTaskSelectionId(task) === selectedTaskId);
      if (matched) return matched;
    }
    return recentTaskHistory[0];
  }, [recentTaskHistory, selectedTaskId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [consoleLines]);

  useEffect(() => {
    if (recentTaskHistory.length === 0) {
      if (selectedTaskId !== null) {
        setSelectedTaskId(null);
      }
      if (showRecentTasks) {
        setShowRecentTasks(false);
      }
      if (showTaskDetails) {
        setShowTaskDetails(false);
      }
      return;
    }
    if (!selectedTaskId) {
      setSelectedTaskId(getTaskSelectionId(recentTaskHistory[0]));
      return;
    }
    const exists = recentTaskHistory.some((task) => getTaskSelectionId(task) === selectedTaskId);
    if (!exists) {
      setSelectedTaskId(getTaskSelectionId(recentTaskHistory[0]));
    }
  }, [recentTaskHistory, selectedTaskId]);

  useEffect(() => {
    if (!installProgress || !installProgress.taskId) {
      installProgressTaskRef.current = null;
      installProgressSectionRef.current = '';
      setDisplayInstallProgressPercent(0);
      return;
    }
    if (installProgressTaskRef.current !== installProgress.taskId) {
      installProgressTaskRef.current = installProgress.taskId;
      installProgressSectionRef.current = installProgress.sectionKey;
      setDisplayInstallProgressPercent(installProgress.percent);
      return;
    }
    if (installProgress.sectionKey && installProgress.sectionKey !== installProgressSectionRef.current) {
      installProgressSectionRef.current = installProgress.sectionKey;
      setDisplayInstallProgressPercent(installProgress.percent);
      return;
    }
    if (installProgress.sectionKey) {
      installProgressSectionRef.current = installProgress.sectionKey;
    }
    setDisplayInstallProgressPercent((prev) => Math.max(prev, installProgress.percent));
  }, [installProgress]);

  const handleCopy = () => {
    const text = consoleLines.join('\n');
    navigator.clipboard.writeText(text).catch(() => {});
  };

  const handleTaskRowClick = (task: TaskResultRecord) => {
    const taskId = getTaskSelectionId(task);
    if (selectedTaskId === taskId && showTaskDetails) {
      setShowTaskDetails(false);
      return;
    }
    setSelectedTaskId(taskId);
    setShowRecentTasks(true);
    setShowTaskDetails(true);
  };

  const handleCopyTask = () => {
    if (!selectedTask) return;
    navigator.clipboard.writeText(JSON.stringify(selectedTask, null, 2)).catch(() => {});
  };

  const handleCopyTaskLogs = () => {
    if (!selectedTask?.log_lines || selectedTask.log_lines.length === 0) return;
    navigator.clipboard.writeText(selectedTask.log_lines.join('\n')).catch(() => {});
  };

  const handleExportTaskBundle = () => {
    if (!selectedTask) return;
    const runtimeDef = selectedTask.runtime_id
      ? runtimeDefs.find((item) => item.id === selectedTask.runtime_id) || null
      : null;
    const bundle = {
      schema_version: 'launcher-task-diagnostic-v1',
      exported_at: new Date().toISOString(),
      source: 'launcher.console_page',
      language,
      selected_task_id: getTaskSelectionId(selectedTask),
      runtime: runtimeDef
        ? {
            id: runtimeDef.id,
            name_zh: runtimeDef.name_zh,
            name_en: runtimeDef.name_en,
            preferred_runtime: runtimeDef.preferred_runtime,
            category: runtimeDef.category,
          }
        : {
            id: selectedTask.runtime_id,
          },
      selected_task: selectedTask,
      active_task_state: currentTaskState.task_id === selectedTask.task_id ? currentTaskState : null,
      global_console_excerpt: consoleLines.slice(-200),
    };
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    const taskId = sanitizeFileNamePart(selectedTask.task_id || selectedTask.task_type || 'task');
    const timestamp = formatExportTimestamp(new Date());
    anchor.href = url;
    anchor.download = `launcher-diagnostic-${taskId}-${timestamp}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-full flex flex-col space-y-3 animate-fade-in">
      {currentTaskState.task_type !== 'idle' && (
        <div className="rounded-2xl p-4 space-y-3" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                {language === 'zh' ? currentTaskState.stage_label_zh : currentTaskState.stage_label_en}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {language === 'zh'
                  ? `任务类型：${currentTaskState.task_type}`
                  : `Task type: ${currentTaskState.task_type}`}
                {currentTaskState.runtime_id ? ` · ${currentTaskState.runtime_id}` : ''}
              </div>
              {currentTaskState.error && (
                <div className="text-xs mt-2" style={{ color: 'var(--danger-text)' }}>
                  {currentTaskState.error}
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              {recentTaskStages.length > 0 && (
                <button
                  onClick={() => setShowCurrentTaskStages((value) => !value)}
                  className="btn-interactive text-[10px] px-2 py-1 rounded flex items-center gap-1"
                  style={{ backgroundColor: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border-card)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-input)'}
                >
                  {showCurrentTaskStages ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
                  {showCurrentTaskStages
                    ? (language === 'zh' ? '收起阶段' : 'Hide stages')
                    : (language === 'zh' ? `展开阶段 (${recentTaskStages.length})` : `Show stages (${recentTaskStages.length})`)}
                </button>
              )}
              {isRunning && (
                <button
                  onClick={() => { void stop(); }}
                  disabled={isStoppingTask}
                  className="btn-interactive text-[10px] px-2 py-1 rounded flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    backgroundColor: isStoppingTask ? 'var(--warning-subtle)' : 'var(--danger-subtle)',
                    color: isStoppingTask ? 'var(--warning-text)' : 'var(--danger-text)',
                    border: `1px solid ${isStoppingTask ? 'var(--warning-border)' : 'var(--danger-border)'}`,
                  }}
                  onMouseEnter={(e) => {
                    if (!isStoppingTask) {
                      e.currentTarget.style.backgroundColor = 'var(--danger-border)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = isStoppingTask ? 'var(--warning-subtle)' : 'var(--danger-subtle)';
                  }}
                >
                  <Square size={10} /> {isStoppingTask ? (language === 'zh' ? '停止中…' : 'Stopping…') : t('btn_stop')}
                </button>
              )}
              <TaskStateBadge state={currentTaskState.state} language={language} taskType={currentTaskState.task_type} />
            </div>
          </div>

          {recentTaskStages.length > 0 && showCurrentTaskStages && (
            <div className="space-y-2">
              <div className="text-[11px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                {language === 'zh' ? '最近阶段' : 'Recent stages'}
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto pr-1 custom-scrollbar">
                {recentTaskStages.map((event, index) => (
                  <div
                    key={`${event.task_id}-${event.stage_code}-${index}`}
                    className="rounded-xl p-3"
                    style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                          {language === 'zh' ? event.stage_label_zh : event.stage_label_en}
                        </div>
                        <div className="text-[11px] mt-1" style={{ color: 'var(--text-secondary)' }}>
                          {event.timestamp}
                        </div>
                      </div>
                      <span className="text-[10px] px-2 py-1 rounded" style={{ backgroundColor: 'var(--bg-card)', color: 'var(--text-secondary)', border: '1px solid var(--border-card)' }}>
                        {formatTaskCodeText(event.result_code || event.code || event.state, t)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {recentTaskHistory.length > 0 && (
        <div className="rounded-2xl p-4 space-y-3" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
          <div className="flex items-center justify-between gap-3">
            <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
              {language === 'zh' ? '最近任务' : 'Recent tasks'}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  setShowRecentTasks((value) => {
                    const next = !value;
                    if (!next) {
                      setShowTaskDetails(false);
                    }
                    return next;
                  });
                }}
                className="btn-interactive text-[10px] px-2 py-1 rounded flex items-center gap-1"
                style={{ backgroundColor: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border-card)' }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-input)'}
              >
                {showRecentTasks ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
                {showRecentTasks
                  ? (language === 'zh' ? '收起任务区' : 'Hide tasks')
                  : (language === 'zh' ? `展开任务区 (${recentTaskHistory.length})` : `Show tasks (${recentTaskHistory.length})`)}
              </button>
            {selectedTask && showTaskDetails && (
              <div className="flex items-center gap-2">
                <button
                  onClick={handleExportTaskBundle}
                  className="btn-interactive text-[10px] px-2 py-1 rounded flex items-center gap-1"
                  style={{ backgroundColor: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border-card)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-input)'}
                >
                  <Download size={10} /> {language === 'zh' ? '导出诊断包' : 'Export diagnostic bundle'}
                </button>
                <button
                  onClick={handleCopyTask}
                  className="btn-interactive text-[10px] px-2 py-1 rounded flex items-center gap-1"
                  style={{ backgroundColor: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border-card)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-input)'}
                >
                  <Copy size={10} /> {language === 'zh' ? '复制任务结果' : 'Copy task result'}
                </button>
              </div>
            )}
            </div>
          </div>
          {!showRecentTasks && (
            <div className="rounded-xl px-3 py-2 text-xs" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)', color: 'var(--text-dim)' }}>
              {language === 'zh'
                ? `最近任务已收起，共 ${recentTaskHistory.length} 条。`
                : `${recentTaskHistory.length} recent task(s) hidden.`}
            </div>
          )}
          {showRecentTasks && (
          <div className="space-y-2 max-h-64 overflow-y-auto pr-1 custom-scrollbar">
            {recentTaskHistory.map((task) => (
              <button
                key={task.task_id || `${task.task_type}-${task.finished_at || task.started_at}`}
                type="button"
                onClick={() => handleTaskRowClick(task)}
                className="w-full rounded-xl p-3 text-left btn-interactive"
                style={{
                  backgroundColor: selectedTask && showTaskDetails && getTaskSelectionId(selectedTask) === getTaskSelectionId(task)
                    ? 'var(--bg-card-hover)'
                    : 'var(--bg-input)',
                  border: `1px solid ${
                    selectedTask && showTaskDetails && getTaskSelectionId(selectedTask) === getTaskSelectionId(task)
                      ? 'var(--accent-border)'
                      : 'var(--border-card)'
                  }`,
                }}
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                      {language === 'zh' ? task.stage_label_zh : task.stage_label_en}
                    </div>
                    <div className="text-[11px] mt-1" style={{ color: 'var(--text-secondary)' }}>
                      {(language === 'zh' ? `任务类型：${task.task_type}` : `Task: ${task.task_type}`) + (task.runtime_id ? ` · ${task.runtime_id}` : '')}
                    </div>
                    <div className="text-[11px] mt-1" style={{ color: 'var(--text-dim)' }}>
                      {task.finished_at || task.started_at || '—'}
                      {typeof task.duration_ms === 'number' ? ` · ${Math.max(0.1, task.duration_ms / 1000).toFixed(1)}s` : ''}
                    </div>
                    {task.error && (
                      <div className="text-[11px] mt-1" style={{ color: 'var(--danger-text)' }}>
                        {task.error}
                      </div>
                    )}
                    {task.task_id && currentTaskState.task_id === task.task_id && isTaskStopping(task.task_type, task.state) && (
                      <div className="text-[11px] mt-1" style={{ color: 'var(--warning-text)' }}>
                        {language === 'zh' ? '停止请求已发送，正在等待训练器退出。' : 'Stop requested, waiting for the trainer process to exit.'}
                      </div>
                    )}
                  </div>
                  <TaskResultCodeBadge
                    state={task.state}
                    text={formatTaskBadgeText(task, language, t)}
                    taskType={task.task_type}
                  />
                </div>
              </button>
            ))}
          </div>
          )}

          {showRecentTasks && showTaskDetails && selectedTask && (
            <div className="rounded-2xl p-4 space-y-3" style={{ backgroundColor: 'var(--bg-input)', border: '1px solid var(--border-card)' }}>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {language === 'zh' ? selectedTask.stage_label_zh : selectedTask.stage_label_en}
                  </div>
                  <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                    {(language === 'zh' ? `任务类型：${selectedTask.task_type}` : `Task: ${selectedTask.task_type}`) + (selectedTask.runtime_id ? ` · ${selectedTask.runtime_id}` : '')}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowTaskDetails(false)}
                    className="btn-interactive text-[10px] px-2 py-1 rounded flex items-center gap-1"
                    style={{ backgroundColor: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border-card)' }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card)'}
                  >
                    <ChevronUp size={10} /> {language === 'zh' ? '收起详情' : 'Hide details'}
                  </button>
                  <TaskStateBadge state={selectedTask.state} language={language} taskType={selectedTask.task_type} />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <TaskDetailField
                  label={language === 'zh' ? '开始时间' : 'Started'}
                  value={selectedTask.started_at || '—'}
                />
                <TaskDetailField
                  label={language === 'zh' ? '结束时间' : 'Finished'}
                  value={selectedTask.finished_at || '—'}
                />
                <TaskDetailField
                  label={language === 'zh' ? '耗时' : 'Duration'}
                  value={formatDuration(selectedTask.duration_ms)}
                />
                <TaskDetailField
                  label={language === 'zh' ? '阶段代码' : 'Stage code'}
                  value={selectedTask.stage_code || '—'}
                />
                <TaskDetailField
                  label={language === 'zh' ? '结果代码' : 'Result code'}
                  value={selectedTask.result_code || selectedTask.code || '—'}
                />
                <TaskDetailField
                  label={language === 'zh' ? '任务 ID' : 'Task ID'}
                  value={selectedTask.task_id || '—'}
                />
              </div>

              {selectedTask.error && (
                <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--danger-subtle)', border: '1px solid var(--danger-border)' }}>
                  <div className="text-[11px] uppercase tracking-wider" style={{ color: 'var(--danger-text)' }}>
                    {language === 'zh' ? '错误信息' : 'Error'}
                  </div>
                  <div className="text-xs mt-2 whitespace-pre-wrap break-words" style={{ color: 'var(--danger-text)' }}>
                    {selectedTask.error}
                  </div>
                </div>
              )}

              {selectedTask.commands && selectedTask.commands.length > 0 && (
                <div className="space-y-2">
                  <div className="text-[11px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                    {language === 'zh' ? '已执行命令' : 'Executed commands'}
                  </div>
                  <div className="space-y-2 max-h-72 overflow-y-auto pr-1 custom-scrollbar">
                    {selectedTask.commands.map((command, index) => (
                      <div
                        key={`${command.command_preview}-${index}`}
                        className="rounded-xl p-3"
                        style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="min-w-0 flex-1">
                            <div className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                              {language === 'zh' ? command.label_zh : command.label_en}
                            </div>
                            <div className="text-[11px] mt-1" style={{ color: 'var(--text-secondary)' }}>
                              {language === 'zh'
                                ? `命令 ${command.index}/${command.total} · ${command.command_kind}`
                                : `Command ${command.index}/${command.total} · ${command.command_kind}`}
                            </div>
                          </div>
                          <TaskResultCodeBadge
                            state={command.status}
                            text={formatCommandStateLabel(command.status, language)}
                          />
                        </div>
                        <div className="mt-2 rounded-lg px-3 py-2 font-mono text-[11px] break-all" style={{ backgroundColor: 'var(--bg-input)', color: 'var(--text-primary)' }}>
                          {command.command_preview}
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
                          <TaskDetailField
                            label={language === 'zh' ? '工作目录' : 'Working directory'}
                            value={command.cwd || '—'}
                          />
                          <TaskDetailField
                            label={language === 'zh' ? '耗时' : 'Duration'}
                            value={formatDuration(command.duration_ms)}
                          />
                          <TaskDetailField
                            label={language === 'zh' ? '开始时间' : 'Started'}
                            value={command.started_at || '—'}
                          />
                          <TaskDetailField
                            label={language === 'zh' ? '结束时间' : 'Finished'}
                            value={command.finished_at || '—'}
                          />
                          <TaskDetailField
                            label={language === 'zh' ? '退出码 / PID' : 'Exit code / PID'}
                            value={`${command.exit_code ?? '—'} / ${command.pid ?? '—'}`}
                          />
                        </div>
                        {command.error && (
                          <div className="text-[11px] mt-2 whitespace-pre-wrap break-words" style={{ color: 'var(--danger-text)' }}>
                            {command.error}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {(selectedTask.log_analysis || (selectedTask.log_lines && selectedTask.log_lines.length > 0)) && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-[11px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                      {language === 'zh' ? '执行日志分析' : 'Execution log analysis'}
                    </div>
                    {selectedTask.log_lines && selectedTask.log_lines.length > 0 && (
                      <button
                        onClick={handleCopyTaskLogs}
                        className="btn-interactive text-[10px] px-2 py-1 rounded flex items-center gap-1"
                        style={{ backgroundColor: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border-card)' }}
                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card)'}
                      >
                        <Copy size={10} /> {language === 'zh' ? '复制日志摘录' : 'Copy log excerpt'}
                      </button>
                    )}
                  </div>

                  {selectedTask.log_analysis && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      <TaskDetailField
                        label={language === 'zh' ? '日志行数' : 'Log lines'}
                        value={String(selectedTask.log_analysis.line_count)}
                      />
                      <TaskDetailField
                        label={language === 'zh' ? '警告数' : 'Warnings'}
                        value={String(selectedTask.log_analysis.warning_count)}
                      />
                      <TaskDetailField
                        label={language === 'zh' ? '错误数' : 'Errors'}
                        value={String(selectedTask.log_analysis.error_count)}
                      />
                      <TaskDetailField
                        label={language === 'zh' ? '命中信号' : 'Signals'}
                        value={String(selectedTask.log_analysis.signal_count)}
                      />
                    </div>
                  )}

                  {selectedTask.log_analysis && selectedTask.log_analysis.signals.length > 0 && (
                    <div className="space-y-2">
                      {selectedTask.log_analysis.signals.map((signal, index) => (
                        <div
                          key={`${signal.code}-${index}`}
                          className="rounded-xl p-3"
                          style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}
                        >
                          <div className="flex items-start justify-between gap-4">
                            <div className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                              {language === 'zh' ? signal.title_zh : signal.title_en}
                            </div>
                            <TaskSeverityBadge signal={signal} language={language} />
                          </div>
                          <div className="text-[11px] mt-2 whitespace-pre-wrap break-words font-mono" style={{ color: 'var(--text-secondary)' }}>
                            {signal.matched_line}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {selectedTask.log_analysis && (
                    <div className="grid grid-cols-1 gap-2">
                      <TaskDetailField
                        label={language === 'zh' ? '最后一条警告' : 'Last warning'}
                        value={selectedTask.log_analysis.last_warning || '—'}
                      />
                      <TaskDetailField
                        label={language === 'zh' ? '最后一条错误' : 'Last error'}
                        value={selectedTask.log_analysis.last_error || '—'}
                      />
                    </div>
                  )}

                  {selectedTask.log_lines && selectedTask.log_lines.length > 0 && (
                    <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
                      <div className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                        {language === 'zh' ? '日志摘录' : 'Log excerpt'}
                      </div>
                      <div className="mt-2 rounded-lg px-3 py-2 font-mono text-[11px] whitespace-pre-wrap break-words max-h-72 overflow-y-auto custom-scrollbar" style={{ backgroundColor: 'var(--bg-input)', color: 'var(--text-primary)' }}>
                        {selectedTask.log_lines.join('\n')}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {selectedTask.stages && selectedTask.stages.length > 0 && (
                <div className="space-y-2">
                  <div className="text-[11px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                    {language === 'zh' ? '任务阶段轨迹' : 'Task timeline'}
                  </div>
                  <div className="space-y-2 max-h-72 overflow-y-auto pr-1 custom-scrollbar">
                    {selectedTask.stages.map((event, index) => (
                      <div
                        key={`${event.task_id || 'task'}-${event.stage_code}-${index}`}
                        className="rounded-xl p-3"
                        style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div>
                            <div className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                              {language === 'zh' ? event.stage_label_zh : event.stage_label_en}
                            </div>
                            <div className="text-[11px] mt-1" style={{ color: 'var(--text-secondary)' }}>
                              {event.timestamp}
                            </div>
                          </div>
                          <TaskResultCodeBadge state={event.state} text={formatTaskCodeText(event.result_code || event.code || event.stage_code, t)} />
                        </div>
                        {event.error && (
                          <div className="text-[11px] mt-2 whitespace-pre-wrap break-words" style={{ color: 'var(--danger-text)' }}>
                            {event.error}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="space-y-2">
                <div className="text-[11px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                  {language === 'zh' ? '结构化详情' : 'Structured details'}
                </div>
                {selectedTask.details && Object.keys(selectedTask.details).length > 0 ? (
                  <div className="space-y-2">
                    {Object.entries(selectedTask.details).map(([key, value]) => (
                      <div
                        key={key}
                        className="rounded-xl p-3"
                        style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}
                      >
                        <div className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                          {key}
                        </div>
                        <div className="text-xs mt-2 whitespace-pre-wrap break-words font-mono" style={{ color: 'var(--text-primary)' }}>
                          {formatTaskDetailValue(value)}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-xl p-3 text-xs" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)', color: 'var(--text-dim)' }}>
                    {language === 'zh' ? '这个任务没有额外结构化详情。' : 'This task has no additional structured details.'}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {lastInstallSummary && (
        <div
          className="rounded-2xl p-4 flex items-start justify-between gap-4"
          style={{
            backgroundColor: lastInstallSummary.success ? 'var(--success-subtle)' : 'var(--warning-subtle)',
            border: `1px solid ${lastInstallSummary.success ? 'var(--success-border)' : 'var(--warning-border)'}`,
          }}
        >
          <div className="flex items-start gap-3">
            {lastInstallSummary.success ? (
              <CheckCircle2 size={18} style={{ color: 'var(--success-text)' }} />
            ) : (
              <AlertTriangle size={18} style={{ color: 'var(--warning-text)' }} />
            )}
            <div>
              <div className="text-sm font-semibold" style={{ color: lastInstallSummary.success ? 'var(--success-text)' : 'var(--warning-text)' }}>
                {lastInstallSummary.success
                  ? t('install_summary_success_title', { runtime: installedRuntimeName || lastInstallSummary.runtimeId })
                  : t('install_summary_failed_title', { runtime: installedRuntimeName || lastInstallSummary.runtimeId })}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                {lastInstallSummary.success
                  ? t('install_summary_success_desc')
                  : t('install_summary_failed_desc')}
              </div>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => {
                clearInstallSummary();
                setActivePage(lastInstallSummary.success ? 'launch' : 'runtime');
              }}
              className="btn-interactive px-3 py-2 rounded-lg text-xs flex items-center gap-2"
              style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)', color: 'var(--text-secondary)' }}
            >
              {lastInstallSummary.success ? t('install_summary_open_launch') : t('runtime_selection')}
              <ArrowRight size={14} />
            </button>
          </div>
        </div>
      )}

      {/* Toolbar */}
      <div className="flex justify-between items-center px-1">
        <div className="flex gap-2 items-center text-xs" style={{ color: 'var(--text-muted)' }}>
          <span
            className={`w-2 h-2 rounded-full ${isRunning ? 'animate-pulse' : ''}`}
            style={{ backgroundColor: isStoppingTask ? 'var(--warning-text)' : isRunning ? 'var(--success)' : 'var(--text-dim)' }}
          />
          {isStoppingTask
            ? (language === 'zh' ? '停止中' : 'Stopping')
            : isRunning ? t('status_running') : t('status_stopped')}
        </div>
        <div className="flex gap-2">
          {isRunning && (
            <button
              onClick={() => { void stop(); }}
              disabled={isStoppingTask}
              className="btn-interactive text-[10px] px-2 py-1 rounded flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                backgroundColor: isStoppingTask ? 'var(--warning-subtle)' : 'var(--danger-subtle)',
                color: isStoppingTask ? 'var(--warning-text)' : 'var(--danger-text)',
              }}
              onMouseEnter={(e) => {
                if (!isStoppingTask) {
                  e.currentTarget.style.backgroundColor = 'var(--danger-border)';
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = isStoppingTask ? 'var(--warning-subtle)' : 'var(--danger-subtle)';
              }}
            >
              <Square size={10} /> {isStoppingTask ? (language === 'zh' ? '停止中…' : 'Stopping…') : t('btn_stop')}
            </button>
          )}
          <button
            onClick={handleCopy}
            className="btn-interactive text-[10px] px-2 py-1 rounded flex items-center gap-1"
            style={{ backgroundColor: 'var(--bg-card)', color: 'var(--text-muted)' }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card)'}
          >
            <Copy size={10} /> {t('console_copy')}
          </button>
          <button
            onClick={clearConsole}
            className="btn-interactive text-[10px] px-2 py-1 rounded"
            style={{ backgroundColor: 'var(--danger-subtle)', color: 'var(--danger-text)' }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--danger-border)'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--danger-subtle)'}
          >
            <Trash2 size={10} /> {t('console_clear')}
          </button>
          {recentTaskHistory.length > 0 && (
            <button
              onClick={() => { void clearTaskHistory(); }}
              className="btn-interactive text-[10px] px-2 py-1 rounded"
              style={{ backgroundColor: 'var(--bg-card)', color: 'var(--text-muted)' }}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card)'}
            >
              <Trash2 size={10} /> {language === 'zh' ? '清空任务历史' : 'Clear task history'}
            </button>
          )}
        </div>
      </div>

      {/* Console output */}
      <div
        ref={scrollRef}
        className="flex-1 rounded-2xl p-4 font-mono text-xs overflow-y-auto leading-relaxed shadow-inner custom-scrollbar select-text cursor-text"
        style={{
          backgroundColor: 'var(--console-bg)',
          border: '1px solid var(--border)',
          minHeight: '260px',
          color: 'var(--console-text)',
          WebkitUserSelect: 'text',
          userSelect: 'text',
        }}
      >
        {consoleLines.length === 0 ? (
          <p className="italic" style={{ color: 'var(--console-muted)' }}>{t('console_empty')}</p>
        ) : (
          consoleLines.map((line, i) => (
            <Line key={i} text={line} />
          ))
        )}
      </div>

      {installProgress && (
        <div className="rounded-2xl p-4 space-y-3" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                {t('install_progress_title')}
                {installProgress.runtimeName ? ` · ${installProgress.runtimeName}` : ''}
              </div>
              <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {installProgress.stageLabel}
              </div>
            </div>
            <div className="text-sm font-semibold" style={{ color: 'var(--accent-text)' }}>
              {Math.round(displayInstallProgressPercent)}%
            </div>
          </div>

          <div className="h-2.5 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-input)' }}>
            <div
              className="h-full transition-all duration-300"
              style={{ width: `${displayInstallProgressPercent}%`, backgroundColor: 'var(--accent)' }}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <TaskDetailField
              label={t('install_progress_phase')}
              value={installProgress.phaseLabel}
            />
            <TaskDetailField
              label={t('install_progress_script', {
                current: String(installProgress.scriptCurrent),
                total: String(installProgress.scriptTotal),
              })}
              value={installProgress.itemLabel || '—'}
            />
            <TaskDetailField
              label={t('install_progress_eta')}
              value={installProgress.etaLabel}
            />
            <TaskDetailField
              label={t('install_progress_downloaded')}
              value={installProgress.downloadedLabel || '—'}
            />
          </div>

          {installProgress.speedLabel && (
            <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              {t('install_progress_speed')}: {installProgress.speedLabel}
            </div>
          )}

          {installProgress.compileHint && (
            <div className="rounded-xl px-3 py-2 text-xs" style={{ backgroundColor: 'var(--warning-subtle)', border: '1px solid var(--warning-border)', color: 'var(--warning-text)' }}>
              {installProgress.compileHint}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TaskStateBadge({ state, language, taskType }: { state: string; language: string; taskType?: string }) {
  const config = isTaskStopping(taskType, state)
    ? {
        label: language === 'zh' ? '停止中' : 'Stopping',
        bg: 'var(--warning-subtle)',
        text: 'var(--warning-text)',
        border: 'var(--warning-border)',
      }
    : state === 'succeeded'
    ? {
        label: language === 'zh' ? '已完成' : 'Succeeded',
        bg: 'var(--success-subtle)',
        text: 'var(--success-text)',
        border: 'var(--success-border)',
      }
    : state === 'interrupted'
      ? {
          label: language === 'zh' ? '已中断' : 'Interrupted',
          bg: 'var(--warning-subtle)',
          text: 'var(--warning-text)',
          border: 'var(--warning-border)',
        }
    : state === 'failed'
      ? {
          label: language === 'zh' ? '失败' : 'Failed',
          bg: 'var(--danger-subtle)',
          text: 'var(--danger-text)',
          border: 'var(--danger-border)',
        }
      : state === 'running'
        ? {
            label: language === 'zh' ? '运行中' : 'Running',
            bg: 'var(--accent-subtle)',
            text: 'var(--accent-text)',
            border: 'var(--accent-border)',
          }
        : {
            label: language === 'zh' ? '准备中' : 'Pending',
            bg: 'var(--warning-subtle)',
            text: 'var(--warning-text)',
            border: 'var(--warning-border)',
          };

  return (
    <span
      className="text-[10px] px-2 py-1 rounded-full whitespace-nowrap"
      style={{ backgroundColor: config.bg, color: config.text, border: `1px solid ${config.border}` }}
    >
      {config.label}
    </span>
  );
}

function TaskResultCodeBadge({ state, text, taskType }: { state: string; text: string; taskType?: string }) {
  const tone = isTaskStopping(taskType, state)
    ? {
        bg: 'var(--warning-subtle)',
        text: 'var(--warning-text)',
        border: 'var(--warning-border)',
      }
    : state === 'succeeded'
    ? {
        bg: 'var(--success-subtle)',
        text: 'var(--success-text)',
        border: 'var(--success-border)',
      }
    : state === 'running'
      ? {
          bg: 'var(--accent-subtle)',
          text: 'var(--accent-text)',
          border: 'var(--accent-border)',
        }
    : state === 'pending'
      ? {
          bg: 'var(--warning-subtle)',
          text: 'var(--warning-text)',
          border: 'var(--warning-border)',
        }
    : state === 'interrupted'
      ? {
          bg: 'var(--warning-subtle)',
          text: 'var(--warning-text)',
          border: 'var(--warning-border)',
        }
      : {
          bg: 'var(--danger-subtle)',
          text: 'var(--danger-text)',
          border: 'var(--danger-border)',
        };

  return (
    <span
      className="text-[10px] px-2 py-1 rounded-full whitespace-nowrap"
      style={{ backgroundColor: tone.bg, color: tone.text, border: `1px solid ${tone.border}` }}
    >
      {text}
    </span>
  );
}

function TaskSeverityBadge({ signal, language }: { signal: TaskLogSignal; language: string }) {
  const state = signal.severity === 'error' ? 'failed' : signal.severity === 'warning' ? 'interrupted' : 'pending';
  const text = signal.severity === 'error'
    ? (language === 'zh' ? '错误' : 'Error')
    : signal.severity === 'warning'
      ? (language === 'zh' ? '警告' : 'Warning')
      : (language === 'zh' ? '信息' : 'Info');
  return <TaskResultCodeBadge state={state} text={text} />;
}

function TaskDetailField({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-card)' }}>
      <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
        {label}
      </div>
      <div className="text-xs mt-2 break-words" style={{ color: 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  );
}

function getTaskSelectionId(task: TaskResultRecord): string {
  return task.task_id || `${task.task_type}-${task.finished_at || task.started_at || 'unknown'}`;
}

function isTaskStopping(taskType: string | undefined, state: string | undefined): boolean {
  return taskType === 'stop' && (state === 'pending' || state === 'running');
}

function buildLiveTaskRecord(
  taskState: {
    task_id: string | null;
    task_type: string;
    state: string;
    runtime_id: string | null;
    stage_code: string;
    stage_label_zh: string;
    stage_label_en: string;
    started_at: string | null;
    finished_at: string | null;
    code?: string | null;
    result_code?: string | null;
    error?: string | null;
    details?: Record<string, unknown>;
  },
  taskStages: TaskResultRecord['stages'] | undefined,
  consoleLogLines: string[],
): TaskResultRecord | null {
  if (!taskState.task_id || taskState.task_type === 'idle') {
    return null;
  }
  return {
    task_id: taskState.task_id,
    task_type: taskState.task_type,
    runtime_id: taskState.runtime_id,
    state: taskState.state as TaskResultRecord['state'],
    stage_code: taskState.stage_code,
    stage_label_zh: taskState.stage_label_zh,
    stage_label_en: taskState.stage_label_en,
    started_at: taskState.started_at,
    finished_at: taskState.finished_at,
    duration_ms: null,
    code: taskState.code || null,
    result_code: taskState.result_code || null,
    error: taskState.error || null,
    details: taskState.details || {},
    stages: Array.isArray(taskStages) ? taskStages : [],
    log_lines: consoleLogLines.slice(-200),
  };
}

function formatTaskBadgeText(task: TaskResultRecord, language: string, t: (key: string) => string): string {
  if (isTaskStopping(task.task_type, task.state)) {
    return language === 'zh' ? '停止中' : 'Stopping';
  }
  return formatTaskCodeText(task.result_code || task.code || task.state, t);
}

function formatTaskCodeText(code: string, t: (key: string) => string): string {
  const translated = t(code);
  return translated === code ? code : translated;
}

function formatDuration(durationMs: number | null | undefined): string {
  if (typeof durationMs !== 'number') return '—';
  return `${Math.max(0.1, durationMs / 1000).toFixed(1)}s`;
}

function formatBytes(bytes: number, language: string): string {
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

function formatSpeed(bytesPerSecond: number | null | undefined, language: string): string | null {
  if (!bytesPerSecond || bytesPerSecond <= 0) {
    return null;
  }
  return `${formatBytes(bytesPerSecond, language)}/s`;
}

function formatEta(seconds: number | null | undefined, language: string, t: (key: string) => string): string {
  if (seconds == null) {
    return t('install_progress_eta_calculating');
  }
  if (seconds <= 0) {
    return language === 'zh' ? '0 秒' : '0s';
  }
  if (seconds < 60) {
    return language === 'zh' ? `${seconds} 秒` : `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainSeconds = seconds % 60;
  if (minutes < 60) {
    return language === 'zh' ? `${minutes} 分 ${remainSeconds} 秒` : `${minutes}m ${remainSeconds}s`;
  }
  const hours = Math.floor(minutes / 60);
  const remainMinutes = minutes % 60;
  return language === 'zh' ? `${hours} 小时 ${remainMinutes} 分` : `${hours}h ${remainMinutes}m`;
}

function inferInstallPhaseFromStage(stageCode: string): string {
  if (stageCode === 'runtime_install.validating_runtime' || stageCode === 'runtime_install.building_plan') {
    return 'preparing';
  }
  if (stageCode === 'runtime_install.executing_scripts' || stageCode === 'runtime_install.running_script') {
    return 'install';
  }
  if (stageCode === 'runtime_install.completed') {
    return 'finalizing';
  }
  return 'preparing';
}

function getInstallPhaseRank(phase: string): number {
  if (phase === 'download') return 0.28;
  if (phase === 'install') return 0.58;
  if (phase === 'compile') return 0.82;
  if (phase === 'finalizing') return 0.96;
  return 0.12;
}

function buildInstallProgressState(
  taskState: {
    task_id: string | null;
    task_type: string;
    state: string;
    runtime_id: string | null;
    stage_code: string;
    stage_label_zh: string;
    stage_label_en: string;
    details?: Record<string, unknown>;
  },
  runtimeDefs: Array<{ id: string; name_zh: string; name_en: string }>,
  language: string,
  t: (key: string, params?: Record<string, string>) => string,
) {
  if (taskState.task_type !== 'install') {
    return null;
  }

  const details = taskState.details || {};
  const runtimeName = (() => {
    if (!taskState.runtime_id) return null;
    const matched = runtimeDefs.find((item) => item.id === taskState.runtime_id);
    if (!matched) return taskState.runtime_id;
    return language === 'zh' ? matched.name_zh : matched.name_en;
  })();

  const scriptTotal = Math.max(1, Number(details.script_total || 1));
  const scriptIndex = Math.min(scriptTotal, Math.max(1, Number(details.script_index || (taskState.stage_code === 'runtime_install.running_script' ? 1 : 0))));
  const phase = String(details.progress_phase || '').trim() || inferInstallPhaseFromStage(taskState.stage_code);
  const phaseRank = getInstallPhaseRank(phase);
  const scriptFraction = taskState.stage_code === 'runtime_install.running_script'
    ? ((Math.max(0, scriptIndex - 1) + phaseRank) / scriptTotal)
    : 0;
  const sectionStartPercent = Number(details.progress_section_start_percent || 0);
  const sectionEndPercent = Number(details.progress_section_end_percent || 0);
  const sectionItemPercent = Number(details.progress_item_percent || 0);
  const sectionKey = String(details.progress_section_key || '').trim();

  let percent = 0;
  if (taskState.state === 'succeeded' || taskState.stage_code === 'runtime_install.completed') {
    percent = 100;
  } else if (taskState.stage_code === 'runtime_install.request_received') {
    percent = 2;
  } else if (taskState.stage_code === 'runtime_install.validating_runtime') {
    percent = 8;
  } else if (taskState.stage_code === 'runtime_install.building_plan') {
    percent = 15;
  } else if (taskState.stage_code === 'runtime_install.executing_scripts') {
    percent = 20;
  } else if (taskState.stage_code === 'runtime_install.running_script') {
    if (sectionEndPercent > sectionStartPercent) {
      const normalizedSectionProgress = Math.max(0, Math.min(100, sectionItemPercent)) / 100;
      const dynamicSectionPercent = sectionStartPercent + ((sectionEndPercent - sectionStartPercent) * normalizedSectionProgress);
      const fallbackPercent = 20 + scriptFraction * 76;
      percent = Math.min(96, Math.max(sectionStartPercent, Math.max(dynamicSectionPercent, fallbackPercent * 0.35)));
    } else {
      percent = Math.min(96, 20 + scriptFraction * 76);
    }
  } else if (taskState.state === 'failed') {
    percent = Math.min(96, 20 + scriptFraction * 76);
  }

  const downloadedBytes = Number(details.progress_downloaded_bytes || 0);
  const totalBytes = Number(details.progress_total_bytes || 0);
  const speedBytesPerSec = Number(details.progress_speed_bytes_per_sec || 0);
  const itemLabel = (language === 'zh'
    ? String(details.progress_item_label_zh || details.command_label_zh || '')
    : String(details.progress_item_label_en || details.command_label_en || '')
  ).trim();
  const phaseLabel = (language === 'zh'
    ? String(details.progress_phase_label_zh || '')
    : String(details.progress_phase_label_en || '')
  ).trim() || t(`install_progress_phase_${phase || 'preparing'}`);

  let etaLabel = t('install_progress_eta_unknown');
  if (phase === 'compile') {
    etaLabel = t('install_progress_eta_calculating');
  } else if (totalBytes > 0 && downloadedBytes >= 0 && speedBytesPerSec > 0) {
    etaLabel = formatEta(Math.max(0, Math.round((totalBytes - downloadedBytes) / speedBytesPerSec)), language, t);
  } else if (taskState.state === 'succeeded') {
    etaLabel = language === 'zh' ? '0 秒' : '0s';
  } else {
    etaLabel = t('install_progress_eta_calculating');
  }

  return {
    taskId: taskState.task_id,
    sectionKey,
    runtimeName,
    stageLabel: language === 'zh' ? taskState.stage_label_zh : taskState.stage_label_en,
    phaseLabel,
    scriptCurrent: scriptIndex,
    scriptTotal,
    itemLabel,
    downloadedLabel: totalBytes > 0 ? `${formatBytes(downloadedBytes, language)} / ${formatBytes(totalBytes, language)}` : '',
    speedLabel: formatSpeed(speedBytesPerSec, language) || '',
    etaLabel,
    compileHint: phase === 'compile' ? t('install_progress_compile_hint') : '',
    percent,
  };
}

function formatTaskDetailValue(value: unknown): string {
  if (value === null || value === undefined) return 'null';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function formatCommandStateLabel(state: TaskCommandRecord['status'], language: string): string {
  if (state === 'succeeded') return language === 'zh' ? '成功' : 'Succeeded';
  if (state === 'running') return language === 'zh' ? '运行中' : 'Running';
  if (state === 'pending') return language === 'zh' ? '等待中' : 'Pending';
  if (state === 'interrupted') return language === 'zh' ? '已中断' : 'Interrupted';
  return language === 'zh' ? '失败' : 'Failed';
}

function sanitizeFileNamePart(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') || 'task';
}

function formatExportTimestamp(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  return `${year}${month}${day}-${hours}${minutes}${seconds}`;
}

function Line({ text }: { text: string }) {
  const segments = colorizeLine(text);
  return (
    <p className="mb-1">
      {segments.map((seg, i) => (
        <span key={i} style={seg.style}>{seg.text}</span>
      ))}
    </p>
  );
}
